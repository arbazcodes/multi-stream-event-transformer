"""
Tests for the multi-stream transformer model.
"""

import pytest
import torch
from modeling.models.model import Model, ModelTrainer, PositionalEncoding, causal_attention_mask
from data.events_loader import EventsLoader


class TestPositionalEncoding:
    """Test positional encoding functionality."""
    
    def test_positional_encoding_shape(self):
        """Test that positional encoding produces correct output shape."""
        d_model = 64
        max_len = 100
        batch_size = 2
        seq_len = 50
        
        pos_enc = PositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_enc(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        d_model = 64
        pos_enc = PositionalEncoding(d_model, dropout=0.0)  # No dropout for deterministic test
        
        x = torch.randn(1, 10, d_model)
        output1 = pos_enc(x)
        output2 = pos_enc(x)
        
        # Should be the same (no dropout)
        torch.testing.assert_close(output1, output2)


class TestCausalMask:
    """Test causal attention mask functionality."""
    
    def test_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        seq_len = 10
        mask = causal_attention_mask(seq_len)
        assert mask.shape == (seq_len, seq_len)
    
    def test_causal_mask_properties(self):
        """Test causal mask has correct properties."""
        seq_len = 5
        mask = causal_attention_mask(seq_len)
        
        # Lower triangle should be False (not masked)
        for i in range(seq_len):
            for j in range(i + 1):
                assert not mask[i, j].item(), f"Position ({i}, {j}) should not be masked"
        
        # Upper triangle should be True (masked)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j].item(), f"Position ({i}, {j}) should be masked"


class TestModel:
    """Test the main model functionality."""
    
    @pytest.fixture
    def model_config(self):
        """Provide model configuration for testing."""
        category_names = [
            'EventType', 'ActorRecordId', 'RecipientRecordId', 'LocationRecordId', 
            'CostCodeRecordId', 'JobCodeRecordId', 'ApprovedByManagerUserRecordId', 
            'IsTimesheet', 'HoursWorked', 'Time_Event_Month', 'Time_Event_Day', 
            'Time_Event_Hour', 'Time_Event_Minute', 'Time_Reference_Month', 
            'Time_Reference_Day', 'Time_Reference_Hour', 'Time_Reference_Minute'
        ]
        return {
            'd_model': 64,
            'n_heads': 4,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'd_categories': [10, 100, 100, 50, 50, 100, 4, 100, 14, 33, 25, 61, 14, 33, 25, 61, 61],  # Match length
            'encoders': ['EventType', 'RecipientRecordId', 'LocationRecordId', 'CostCodeRecordId', 'JobCodeRecordId'],
            'category_names': category_names
        }
    
    def test_model_initialization(self, model_config):
        """Test model initializes correctly."""
        model = Model(**model_config)
        assert model.d_model == model_config['d_model']
        assert model.n_heads == model_config['n_heads']
        assert len(model.encoders) == len(model_config['encoders'])
    
    def test_model_forward_pass(self, model_config):
        """Test model forward pass works."""
        model = Model(**model_config)
        
        # Create sample input
        src_seq_len = 20
        tgt_seq_len = 30
        
        src_events = {}
        for name in model_config['encoders']:
            idx = model_config['category_names'].index(name)
            vocab_size = model_config['d_categories'][idx]
            src_events[name] = torch.randint(0, vocab_size, (src_seq_len,))
        
        # Create target events with proper vocabulary sizes for each field
        tgt_columns = []
        for i, vocab_size in enumerate(model_config['d_categories']):
            col = torch.randint(0, vocab_size, (tgt_seq_len,))
            tgt_columns.append(col)
        tgt_events = torch.stack(tgt_columns, dim=1)
        
        masks = [causal_attention_mask(tgt_seq_len)]
        
        # Forward pass
        loss, per_field_losses = model(src_events, tgt_events, masks)
        
        assert isinstance(loss, torch.Tensor)
        assert len(per_field_losses) == len(model_config['category_names'])
        assert loss.item() > 0  # Loss should be positive


class TestModelTrainer:
    """Test the model trainer functionality."""
    
    def test_model_trainer_initialization(self):
        """Test model trainer initializes correctly."""
        trainer = ModelTrainer(
            d_input=100,
            d_model=64,
            n_heads=4,
            encoder_layers=2,
            decoder_layers=2,
            d_categories=[10, 100, 100, 50, 50, 100, 4, 100, 14, 33, 25, 61, 14, 33, 25, 61, 61],  # 17 categories
            encoders=['EventType', 'RecipientRecordId', 'LocationRecordId', 'CostCodeRecordId', 'JobCodeRecordId'],
            d_output=100
        )
        
        assert trainer.model is not None
        assert len(trainer.category_names) == 17  # 5 encoders + 12 additional fields
    
    def test_model_info(self):
        """Test model info generation."""
        trainer = ModelTrainer(
            d_input=100,
            d_model=64,
            n_heads=4,
            encoder_layers=2,
            decoder_layers=2,
            d_categories=[10, 100, 100, 50, 50, 100, 4, 100, 14, 33, 25, 61, 14, 33, 25, 61, 61],  # 17 categories
            encoders=['EventType', 'RecipientRecordId', 'LocationRecordId', 'CostCodeRecordId', 'JobCodeRecordId'],
            d_output=100
        )
        
        info = trainer.get_model_info()
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0


class TestEventsLoader:
    """Test the events data loader."""
    
    def test_events_loader_initialization(self):
        """Test events loader initializes correctly."""
        loader = EventsLoader(
            batch_size=8,
            src_seq_length=50,
            tgt_seq_length=100,
            id_category_size=500,
            dataset_size=100
        )
        
        assert len(loader) == 100
        assert loader.src_sequence_length == 50
        assert loader.tgt_sequence_length == 100
    
    def test_events_loader_getitem(self):
        """Test events loader returns correct data format."""
        loader = EventsLoader(
            batch_size=8,
            src_seq_length=20,
            tgt_seq_length=30,
            id_category_size=100,
            dataset_size=10
        )
        
        target, source, masks = loader[0]
        
        # Check shapes
        assert target.shape == (30, len(loader.category_fields))
        assert len(source) == len(loader.encoder_streams)
        assert len(masks) == 1
        assert masks[0].shape == (30, 30)
        
        # Check data types
        assert target.dtype == torch.long
        for stream_data in source.values():
            assert stream_data.dtype == torch.long
        assert masks[0].dtype == torch.bool
    
    def test_events_loader_category_info(self):
        """Test category info retrieval."""
        loader = EventsLoader(
            batch_size=8,
            src_seq_length=20,
            tgt_seq_length=30,
            id_category_size=100,
            dataset_size=10
        )
        
        category_info = loader.get_category_info()
        encoder_info = loader.get_encoder_info()
        
        assert isinstance(category_info, dict)
        assert isinstance(encoder_info, list)
        assert len(encoder_info) == 5  # 5 encoder streams


if __name__ == "__main__":
    pytest.main([__file__])