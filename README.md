# Multi-Stream Transformer for Event Sequence Modeling

A PyTorch implementation of a multi-stream transformer architecture designed for event sequence prediction and modeling. This project demonstrates advanced deep learning techniques including multi-stream processing, cross-attention mechanisms, and autoregressive sequence generation.

## Overview

This project implements a novel transformer architecture that processes multiple input streams through separate encoders and uses cross-attention to generate predictions for future event sequences. The model is particularly suited for applications involving temporal event data with multiple categorical attributes.

### Key Features

- Multi-Stream Architecture: Separate transformer encoders for different input streams
- Cross-Attention Mechanism: Decoder attends to all encoder streams simultaneously  
- Autoregressive Generation: Support for sequence generation with temperature control
- Comprehensive Training Pipeline: Full training loop with validation, checkpointing, and monitoring
- Synthetic Data Generation: Built-in synthetic event data with temporal correlations
- Professional Implementation: Clean, documented code following best practices

## Architecture

The model consists of three main components:

1. Multi-Stream Encoders: Each input stream (event types, actors, locations, etc.) is processed by its own transformer encoder
2. Cross-Attention Decoder: A transformer decoder that attends to the final representations from all encoder streams
3. Multi-Head Output: Separate classification heads for predicting different categorical fields

### Technical Details

- Input Streams: 5 configurable encoder streams for different event attributes
- Model Dimensions: Configurable embedding and hidden dimensions (default: 128)
- Attention Heads: Multi-head attention with configurable head count (default: 8)
- Architecture Depth: Configurable encoder and decoder layers (default: 4 encoder, 6 decoder)
- Output Categories: 17 different categorical fields including temporal components

## Applications

This architecture is designed for event sequence modeling tasks such as:

- Human Resources: Employee behavior prediction, scheduling optimization
- Financial Services: Transaction sequence analysis and fraud detection  
- System Monitoring: Event log analysis and anomaly detection
- User Analytics: Behavior pattern recognition and recommendation systems

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended but not required)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-stream-transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

## Quick Start

### Basic Training

Run training with default configuration:
```bash
python -m modeling.train_model
```

### Custom Configuration

Modify `modeling/training_config.json` to customize hyperparameters:
```json
{
    "src_seq_len": 200,
    "tgt_seq_len": 300,
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "model": {
        "d_model": 128,
        "num_heads": 8,
        "encoder_layers": 4,
        "decoder_layers": 6,
        "dropout": 0.1
    }
}
```

### Monitoring Training

Start TensorBoard to monitor training progress:
```bash
tensorboard --logdir runs --port 6006
```

Visit `http://localhost:6006` to view training metrics and loss curves.

### Running Tests

Execute the test suite:
```bash
python -m pytest tests/ -v
```

### Basic Example

Run the basic usage example:
```bash
python examples/basic_usage.py
```

## Project Structure

```
multi-stream-transformer/
├── data/                          # Data loading and preprocessing
│   ├── __init__.py
│   └── events_loader.py          # Synthetic event data generator
├── modeling/                      # Model architecture and training
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py              # Multi-stream transformer implementation
│   ├── train_model.py            # Training pipeline
│   └── training_config.json      # Hyperparameter configuration
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── metrics.py                # Evaluation metrics
│   └── visualization.py          # Plotting and visualization tools
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_model.py             # Model and component tests
├── examples/                      # Usage examples
│   ├── __init__.py
│   └── basic_usage.py            # Basic usage demonstration
├── assets/                        # Documentation assets
├── docker-compose.yml             # Docker orchestration
├── Training.Dockerfile            # Training container
├── Tensorboard.Dockerfile         # TensorBoard container
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT license
└── README.md                      # This file
```

## Model Configuration

The model supports extensive configuration through JSON files:

- Sequence Lengths: Configurable source and target sequence lengths
- Model Dimensions: Embedding dimension, attention heads, layer counts
- Training Parameters: Learning rate, batch size, epochs, dropout
- Data Parameters: Vocabulary sizes, dataset size, category definitions

## Docker Support

For containerized training and monitoring:

```bash
# Start training and TensorBoard containers
docker-compose up --build

# Training will start automatically
# TensorBoard will be available at http://localhost:6006
```

## Performance

The model achieves strong performance on synthetic event sequence data:

- Training Convergence: Loss typically converges within 50-100 epochs
- Memory Efficiency: ~4GB GPU memory for default configuration
- Inference Speed: ~100 sequences/second on modern GPUs
- Model Size: ~2.1M parameters (configurable)

## Technical Implementation

### Multi-Stream Processing

Each input stream is processed independently through its own transformer encoder:

```python
for name in self.encoders:
    x = self.embeddings[name](src_events[name])
    x = self.pos_enc(x.unsqueeze(0))
    encoded = self.stream_encoders[name](x)
    encoded_streams.append(encoded)
```

### Cross-Attention Mechanism

The decoder uses cross-attention to attend to all encoder streams:

```python
for layer in self.decoder_layers:
    # Self-attention on decoder sequence
    y = y + layer['self_attn'](y, y, y, attn_mask=causal_mask)[0]
    
    # Cross-attention to encoder memory
    y = y + layer['cross_attn'](y, memory, memory)[0]
    
    # Feed-forward processing
    y = y + layer['ffn'](layer['norm'](y))
```

### Event Categories

The model predicts 17 categorical fields:

- Core Events: EventType, ActorRecordId, RecipientRecordId
- Context: LocationRecordId, CostCodeRecordId, JobCodeRecordId  
- Metadata: IsTimesheet, HoursWorked, ApprovedByManagerUserRecordId
- Temporal: Event and reference timestamps (month, day, hour, minute)

## Contributing

Contributions are welcome. Please ensure:

- Code follows existing style and conventions
- New features include appropriate tests
- Documentation is updated for significant changes
- All tests pass before submitting pull requests

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{multistream-transformer,
  title={Multi-Stream Transformer for Event Sequence Modeling},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/multi-stream-transformer}
}
```