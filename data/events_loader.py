"""
Event Data Loader for Multi-Stream Transformer Training

This module provides a PyTorch Dataset implementation for loading and preprocessing
event sequence data for training the multi-stream transformer model.

The dataset generates synthetic event sequences with multiple categorical fields
and temporal information, simulating real-world event data patterns.

Author: [Your Name]
Date: 2024
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np

class EventsLoader(Dataset):
    """
    Dataset loader for event sequence data.
    
    This dataset generates synthetic event sequences with multiple categorical fields
    representing different aspects of events (actors, locations, time, etc.).
    
    Args:
        batch_size: Batch size for training (kept for compatibility)
        src_seq_length: Length of source sequences (encoder input)
        tgt_seq_length: Length of target sequences (decoder output)
        id_category_size: Size of ID-based categories (actors, locations, etc.)
        dataset_size: Total number of samples in the dataset
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        batch_size: int,
        src_seq_length: int,
        tgt_seq_length: int,
        id_category_size: int,
        dataset_size: int = 1000,
        seed: Optional[int] = 42
    ):
        self.batch_size = batch_size
        self.src_sequence_length = src_seq_length
        self.tgt_sequence_length = tgt_seq_length
        self.dataset_size = dataset_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Core event categories that appear in both encoder and decoder
        self.initial_category_fields = {
            'EventType': 10,                    # Different types of events
            'ActorRecordId': id_category_size,  # Person performing the event
            'RecipientRecordId': id_category_size,  # Person receiving the event
            'LocationRecordId': id_category_size,   # Location of the event
            'CostCodeRecordId': id_category_size,   # Cost center/project code
            'JobCodeRecordId': id_category_size,    # Job/role code
            'ApprovedByManagerUserRecordId': id_category_size,  # Approving manager
            'IsTimesheet': 4,                   # Boolean-like field (0-3)
            'HoursWorked': 100,                 # Hours worked (0-99)
        }

        # Extended categories including temporal decomposition
        self.category_fields = {
            **self.initial_category_fields,
            # Event timestamp components
            'Time_Event_Month': 14,      # Months (1-12) + padding
            'Time_Event_Day': 33,        # Days (1-31) + padding
            'Time_Event_Hour': 25,       # Hours (0-23) + padding
            'Time_Event_Minute': 61,     # Minutes (0-59) + padding
            # Reference timestamp components
            'Time_Reference_Month': 14,
            'Time_Reference_Day': 33,
            'Time_Reference_Hour': 25,
            'Time_Reference_Minute': 61,
        }

        # Encoder streams (subset of categories used as input)
        self.encoder_streams = [
            'EventType',
            'RecipientRecordId',
            'LocationRecordId',
            'CostCodeRecordId',
            'JobCodeRecordId',
        ]

        # Computed properties
        self.input_size = len(self.encoder_streams) * self.src_sequence_length
        self.output_size = len(self.category_fields) * self.tgt_sequence_length
        
        print(f"Initialized EventsLoader:")
        print(f"  - Dataset size: {self.dataset_size}")
        print(f"  - Source sequence length: {self.src_sequence_length}")
        print(f"  - Target sequence length: {self.tgt_sequence_length}")
        print(f"  - Encoder streams: {len(self.encoder_streams)}")
        print(f"  - Total categories: {len(self.category_fields)}")

    def load_dataset(self) -> None:
        """
        Load or prepare the dataset.
        
        For synthetic data, this is a no-op. In a real implementation,
        this would load data from files or databases.
        """
        print("Dataset loaded (synthetic data generation)")
        return

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.dataset_size

    def get_category_info(self) -> Dict[str, int]:
        """Get information about category fields and their sizes."""
        return self.category_fields.copy()

    def get_encoder_info(self) -> List[str]:
        """Get information about encoder streams."""
        return self.encoder_streams.copy()

    def _generate_correlated_sequence(
        self, 
        base_values: torch.Tensor, 
        seq_length: int, 
        correlation: float = 0.7
    ) -> torch.Tensor:
        """
        Generate a sequence with some correlation to base values.
        
        This creates more realistic synthetic data by introducing
        temporal dependencies and patterns.
        """
        sequence = torch.zeros(seq_length, dtype=torch.long)
        
        for i in range(seq_length):
            if i == 0 or torch.rand(1) > correlation:
                # Random selection
                sequence[i] = torch.randint(0, len(base_values), (1,))
            else:
                # Correlated with previous value
                prev_idx = sequence[i-1].item()
                # Small perturbation around previous value
                new_idx = prev_idx + torch.randint(-2, 3, (1,)).item()
                new_idx = max(0, min(len(base_values) - 1, new_idx))
                sequence[i] = new_idx
                
        return sequence

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Generate a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (target_events, source_events, masks)
            - target_events: Tensor of shape (tgt_seq_len, num_fields)
            - source_events: Dict mapping stream names to tensors of shape (src_seq_len,)
            - masks: List containing causal attention mask
        """
        # Set seed based on index for reproducibility
        torch.manual_seed(idx + 42)
        
        # 1) Generate target sequences for all categories
        target_columns = []
        for name, vocab_size in self.category_fields.items():
            if 'Time_' in name:
                # Generate more realistic temporal patterns
                if 'Month' in name:
                    col = torch.randint(1, 13, (self.tgt_sequence_length,))  # 1-12
                elif 'Day' in name:
                    col = torch.randint(1, 32, (self.tgt_sequence_length,))  # 1-31
                elif 'Hour' in name:
                    col = torch.randint(0, 24, (self.tgt_sequence_length,))  # 0-23
                elif 'Minute' in name:
                    col = torch.randint(0, 60, (self.tgt_sequence_length,))  # 0-59
                else:
                    col = torch.randint(0, vocab_size, (self.tgt_sequence_length,))
            else:
                # Generate sequences with some temporal correlation
                base_values = torch.arange(vocab_size)
                col = self._generate_correlated_sequence(
                    base_values, self.tgt_sequence_length, correlation=0.6
                )
                # Ensure values are within vocabulary range
                col = torch.clamp(col, 0, vocab_size - 1)
            
            target_columns.append(col)

        # Stack into target tensor: (tgt_seq_len, num_fields)
        target_events = torch.stack(target_columns, dim=1)

        # 2) Generate source sequences for encoder streams
        source_events = {}
        for stream_name in self.encoder_streams:
            vocab_size = self.category_fields[stream_name]
            
            # Generate correlated sequences for more realistic patterns
            base_values = torch.arange(vocab_size)
            sequence = self._generate_correlated_sequence(
                base_values, self.src_sequence_length, correlation=0.8
            )
            sequence = torch.clamp(sequence, 0, vocab_size - 1)
            
            source_events[stream_name] = sequence

        # 3) Generate causal attention mask
        causal_mask = torch.triu(
            torch.ones((self.tgt_sequence_length, self.tgt_sequence_length), dtype=torch.bool),
            diagonal=1
        )

        return target_events, source_events, [causal_mask]

    def get_sample_batch(self, batch_size: int = 4) -> Tuple[List, List, List]:
        """
        Generate a sample batch for testing or demonstration.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Tuple of (targets, sources, masks) lists
        """
        targets, sources, masks = [], [], []
        
        for i in range(batch_size):
            tgt, src, mask = self[i]
            targets.append(tgt)
            sources.append(src)
            masks.append(mask)
            
        return targets, sources, masks

    def print_sample_info(self, idx: int = 0) -> None:
        """Print information about a sample for debugging."""
        target, source, masks = self[idx]
        
        print(f"Sample {idx} Information:")
        print(f"  Target shape: {target.shape}")
        print(f"  Source streams: {list(source.keys())}")
        for name, tensor in source.items():
            print(f"    {name}: {tensor.shape}")
        print(f"  Mask shape: {masks[0].shape}")
        print(f"  Target categories: {list(self.category_fields.keys())}")
        print(f"  Sample target values (first 5 timesteps):")
        for i, name in enumerate(self.category_fields.keys()):
            if i < target.shape[1]:
                values = target[:5, i].tolist()
                print(f"    {name}: {values}")

# Example usage and testing
if __name__ == "__main__":
    # Create dataset
    dataset = EventsLoader(
        batch_size=8,
        src_seq_length=50,
        tgt_seq_length=100,
        id_category_size=500,
        dataset_size=100
    )
    
    dataset.load_dataset()
    
    # Print dataset info
    print(f"\nDataset length: {len(dataset)}")
    dataset.print_sample_info(0)
    
    # Test batch generation
    targets, sources, masks = dataset.get_sample_batch(batch_size=3)
    print(f"\nBatch test:")
    print(f"  Batch size: {len(targets)}")
    print(f"  First target shape: {targets[0].shape}")
    print(f"  First source keys: {list(sources[0].keys())}")
