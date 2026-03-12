"""
Multi-Stream Transformer Architecture for Event Sequence Modeling

This module implements a sophisticated transformer-based architecture that processes
multiple input streams through separate encoders and uses cross-attention to generate
predictions for future event sequences.

Key Features:
- Multi-stream encoder processing
- Cross-attention mechanism
- Causal masking for autoregressive generation
- Configurable architecture parameters
- Support for categorical event prediction

Author: [Your Name]
Date: 2024
"""

from typing import List, Dict, Optional, Tuple
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ModuleDict, ModuleList
import math

def causal_attention_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate a causal attention mask for autoregressive generation.
    
    Args:
        seq_len: Length of the sequence
        device: Device to place the mask on
        
    Returns:
        Boolean mask tensor of shape (seq_len, seq_len) where True indicates
        positions that should be masked (future positions)
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    This implementation uses sine and cosine functions of different frequencies
    to encode position information that can be added to token embeddings.
    
    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length to support
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class Model(nn.Module):
    """
    Multi-Stream Transformer Architecture for Event Sequence Modeling.
    
    This model processes multiple input streams through separate transformer encoders
    and uses cross-attention to generate predictions for future event sequences.
    
    Architecture:
    1. Multiple input streams are processed by separate transformer encoders
    2. The last hidden state from each encoder is used as memory for cross-attention
    3. A decoder with self-attention and cross-attention layers generates predictions
    4. Multiple output heads predict different categorical fields
    
    Args:
        d_model: Dimension of model embeddings and hidden states
        n_heads: Number of attention heads
        encoder_layers: Number of layers in each stream encoder
        decoder_layers: Number of decoder layers
        d_categories: List of vocabulary sizes for each category
        encoders: List of encoder stream names
        category_names: List of all category names (encoders + additional fields)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        d_categories: List[int],
        encoders: List[str],
        category_names: List[str],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoders = encoders
        self.category_names = category_names
        self.d_model = d_model
        self.n_heads = n_heads

        # Input validation
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert len(d_categories) == len(category_names), "Mismatch between categories and names"

        # 1) Embeddings for each category
        self.embeddings = ModuleDict({
            name: nn.Embedding(num_cat, d_model)
            for name, num_cat in zip(category_names, d_categories)
        })

        # 2) TransformerEncoder per input stream
        self.stream_encoders = ModuleDict()
        for name in encoders:
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_heads, 
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            )
            self.stream_encoders[name] = TransformerEncoder(
                encoder_layer, num_layers=encoder_layers
            )

        # 3) Positional encodings
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=2000, dropout=dropout)
        self.pos_dec = PositionalEncoding(d_model=d_model, max_len=2000, dropout=dropout)

        # 4) Decoder layers with residual connections and layer normalization
        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_layers):
            layer = nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            })
            self.decoder_layers.append(layer)

        # 5) Output heads with dropout for regularization
        self.output_dropout = nn.Dropout(dropout)
        self.heads = ModuleDict({
            name: nn.Linear(d_model, num_cat)
            for name, num_cat in zip(category_names, d_categories)
        })

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        src_events: Dict[str, torch.Tensor],
        tgt_events: torch.Tensor,
        masks: List[torch.Tensor],
        run_backward: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the multi-stream transformer model.
        
        Args:
            src_events: Dictionary mapping encoder names to input sequences
                       Each tensor has shape (src_seq_len,)
            tgt_events: Target events tensor of shape (tgt_seq_len, num_fields)
            masks: List containing causal attention masks
            run_backward: Whether to run backward pass (for training)
            
        Returns:
            Tuple of (total_loss, per_field_losses)
        """
        device = next(self.parameters()).device
        
        # --- Encode each input stream separately ---
        encoded_streams = []
        for name in self.encoders:
            if name not in src_events:
                raise ValueError(f"Missing encoder stream: {name}")
                
            x = src_events[name].to(device)  # (src_len,)
            x = self.embeddings[name](x)     # (src_len, d_model)
            x = x.unsqueeze(0)               # (1, src_len, d_model)
            x = self.pos_enc(x)              # (1, src_len, d_model)
            x = self.stream_encoders[name](x)  # (1, src_len, d_model)
            encoded_streams.append(x)

        # --- Build memory: concatenate last tokens from each stream ---
        # Shape: (1, num_streams, d_model)
        memory = torch.stack([stream[:, -1, :] for stream in encoded_streams], dim=1)

        # --- Prepare decoder input ---
        tgt_events = tgt_events.to(device)  # (tgt_len, num_fields)
        
        # Embed and sum each field
        field_embeddings = []
        for i, name in enumerate(self.category_names):
            if i < tgt_events.size(1):
                field_emb = self.embeddings[name](tgt_events[:, i])
                field_embeddings.append(field_emb)
        
        # Sum embeddings and add batch dimension
        y = sum(field_embeddings)            # (tgt_len, d_model)
        y = y.unsqueeze(0)                   # (1, tgt_len, d_model)
        y = self.pos_dec(y)                  # (1, tgt_len, d_model)

        # --- Decoder layers with residual connections ---
        causal_mask = masks[0].to(device)    # (tgt_len, tgt_len)
        
        for layer in self.decoder_layers:
            # Self-attention with residual connection
            y_norm = layer['norm1'](y)
            attn_out, _ = layer['self_attn'](y_norm, y_norm, y_norm, attn_mask=causal_mask)
            y = y + attn_out
            
            # Cross-attention with residual connection
            y_norm = layer['norm2'](y)
            cross_out, _ = layer['cross_attn'](y_norm, memory, memory)
            y = y + cross_out
            
            # Feed-forward with residual connection
            y_norm = layer['norm3'](y)
            ffn_out = layer['ffn'](y_norm)
            y = y + ffn_out

        # Remove batch dimension and apply dropout
        y = y.squeeze(0)                     # (tgt_len, d_model)
        y = self.output_dropout(y)

        # --- Compute losses for each field ---
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        per_field_losses = []
        
        for idx, name in enumerate(self.category_names):
            if idx < tgt_events.size(1):
                logits = self.heads[name](y)     # (tgt_len, vocab_size)
                targets = tgt_events[:, idx]     # (tgt_len,)
                loss = criterion(logits, targets)
                per_field_losses.append(loss)
                total_loss = total_loss + loss

        if run_backward and total_loss.requires_grad:
            total_loss.backward()
            
        return total_loss, per_field_losses

    def generate(
        self, 
        src_events: Dict[str, torch.Tensor], 
        max_length: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate event sequences autoregressively.
        
        Args:
            src_events: Source event streams
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated sequence tensor
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Encode source streams
            encoded_streams = []
            for name in self.encoders:
                x = src_events[name].to(device)
                x = self.embeddings[name](x).unsqueeze(0)
                x = self.pos_enc(x)
                x = self.stream_encoders[name](x)
                encoded_streams.append(x)
            
            memory = torch.stack([stream[:, -1, :] for stream in encoded_streams], dim=1)
            
            # Initialize with start tokens (zeros)
            generated = torch.zeros((1, len(self.category_names)), dtype=torch.long, device=device)
            
            for _ in range(max_length):
                # Create causal mask
                seq_len = generated.size(0)
                causal_mask = causal_attention_mask(seq_len, device)
                
                # Embed current sequence
                field_embeddings = []
                for i, name in enumerate(self.category_names):
                    field_emb = self.embeddings[name](generated[:, i])
                    field_embeddings.append(field_emb)
                
                y = sum(field_embeddings).unsqueeze(0)
                y = self.pos_dec(y)
                
                # Decode
                for layer in self.decoder_layers:
                    y_norm = layer['norm1'](y)
                    attn_out, _ = layer['self_attn'](y_norm, y_norm, y_norm, attn_mask=causal_mask)
                    y = y + attn_out
                    
                    y_norm = layer['norm2'](y)
                    cross_out, _ = layer['cross_attn'](y_norm, memory, memory)
                    y = y + cross_out
                    
                    y_norm = layer['norm3'](y)
                    y = y + layer['ffn'](y_norm)
                
                # Sample next tokens
                y = y.squeeze(0)[-1:]  # Last position only
                next_tokens = []
                
                for name in self.category_names:
                    logits = self.heads[name](y) / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    next_tokens.append(next_token)
                
                next_tokens = torch.cat(next_tokens, dim=1)
                generated = torch.cat([generated, next_tokens], dim=0)
            
            return generated

class ModelTrainer(nn.Module):
    """
    Training wrapper for the multi-stream transformer model.
    
    This class handles batch processing and provides a clean interface
    for training the model with multiple samples per batch.
    
    Args:
        d_input: Input dimension (for compatibility)
        d_model: Model embedding dimension
        n_heads: Number of attention heads
        encoder_layers: Number of encoder layers per stream
        decoder_layers: Number of decoder layers
        d_categories: List of vocabulary sizes for each category
        encoders: List of encoder stream names
        d_output: Output dimension (for compatibility)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        d_categories: List[int],
        encoders: List[str],
        d_output: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Define all category names (encoders + additional prediction fields)
        self.category_names = list(encoders) + [
            'ActorRecordId', 'ApprovedByManagerUserRecordId', 'IsTimesheet', 'HoursWorked',
            'Time_Event_Month', 'Time_Event_Day', 'Time_Event_Hour', 'Time_Event_Minute',
            'Time_Reference_Month', 'Time_Reference_Day', 'Time_Reference_Hour', 'Time_Reference_Minute'
        ]
        
        self.model = Model(
            d_model=d_model,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_categories=d_categories,
            encoders=encoders,
            category_names=self.category_names,
            dropout=dropout
        )
        
        # Optimizer will be set externally
        self.optimizer = None

    def forward(
        self,
        batch_src: List[Dict[str, torch.Tensor]],
        batch_tgt: List[torch.Tensor],
        batch_masks: List[List[torch.Tensor]],
        run_backward: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process a batch of samples through the model.
        
        Args:
            batch_src: List of source event dictionaries
            batch_tgt: List of target event tensors
            batch_masks: List of mask lists for each sample
            run_backward: Whether to run backward pass
            
        Returns:
            Tuple of (average_loss, average_per_field_losses)
        """
        losses = []
        all_per_field = [[] for _ in range(len(self.category_names))]
        
        for src, tgt, masks in zip(batch_src, batch_tgt, batch_masks):
            try:
                loss, per_field = self.model(src, tgt, masks, run_backward)
                losses.append(loss)
                
                # Collect per-field losses
                for i, field_loss in enumerate(per_field):
                    if i < len(all_per_field):
                        all_per_field[i].append(field_loss)
                        
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        if not losses:
            # Return zero losses if no samples processed successfully
            device = next(self.model.parameters()).device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            zero_per_field = [torch.tensor(0.0, device=device) for _ in self.category_names]
            return zero_loss, zero_per_field
        
        # Average losses across batch
        avg_loss = torch.stack(losses).mean()
        avg_per_field = [
            torch.stack(field_losses).mean() if field_losses else torch.tensor(0.0)
            for field_losses in all_per_field
        ]
        
        return avg_loss, avg_per_field

    def get_model_info(self) -> Dict:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'encoder_streams': len(self.model.encoders),
            'category_names': len(self.category_names)
        }
