# Federated Learning with CLIP Alignment

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This project implements a novel federated learning system with CLIP semantic alignment, enabling privacy-preserving, semantically-rich distributed learning. The key innovation is bridging federated learning with large foundation models (CLIP) while preserving privacy.

### Key Features

- **Federated Learning Architecture**: Train models across multiple clients without sharing raw data
- **CLIP Semantic Alignment**: Align local embeddings with CLIP's rich semantic space
- **Privacy Preservation**: Share only model/mapper weights, never raw data or CLIP embeddings
- **Dual Objective Training**: Optimize for both classification accuracy and semantic mapping
- **Differential Privacy Option**: Add calibrated noise for enhanced privacy guarantees

## How It Works

1. **Client-Side**: 
   - CNN extracts compact embeddings (`z_small`) and class predictions
   - Local CLIP model produces rich embeddings (`z_clip`)
   - A mapper network (φ) transforms `z_small` to match CLIP's space
   - Models train with dual loss (classification + alignment)

2. **Server-Side**:
   - Aggregates CNN and mapper weights from all clients
   - Creates global model and distributes back to clients
   - Never sees raw data or CLIP embeddings

3. **Privacy Mechanism**:
   - Only small network weights shared (not raw data)
   - Optional differential privacy noise
   - CLIP model remains local as a "teacher"

## Architecture

```
┌─────────────┐                                 ┌─────────────┐
│   Client 1  │                                 │   Client 2  │
│  ┌───────┐  │                                 │  ┌───────┐  │
│  │ Image │  │                                 │  │ Image │  │
│  └───┬───┘  │                                 │  └───┬───┘  │
│      │      │                                 │      │      │
│  ┌───▼───┐  │                                 │  ┌───▼───┐  │
│  │  CNN  │  │                                 │  │  CNN  │  │
│  └───┬───┘  │                                 │  └───┬───┘  │
│      │      │                                 │      │      │
│  ┌───▼────┐ │                                 │  ┌───▼────┐ │
│  │z_small │ │                                 │  │z_small │ │
│  └─┬────┬─┘ │                                 │  └─┬────┬─┘ │
│    │    │   │                                 │    │    │   │
│┌───▼──┐ │   │                                 │┌───▼──┐ │   │
││ CLIP │ │   │                                 ││ CLIP │ │   │
│└──┬───┘ │   │                                 │└──┬───┘ │   │
│   │     │   │                                 │   │     │   │
│┌──▼───┐ │   │                                 │┌──▼───┐ │   │
││z_clip│ │   │                                 ││z_clip│ │   │
│└──┬───┘ │   │                                 │└──┬───┘ │   │
│   │   ┌─▼─┐ │                                 │   │   ┌─▼─┐ │
│   └──►│ φ │ │                                 │   └──►│ φ │ │
│       └─┬─┘ │                                 │       └─┬─┘ │
│         │   │                                 │         │   │
└─────────┼───┘                                 └─────────┼───┘
          │                                               │
          │               ┌─────────┐                     │
          └──────────────►│ Server  │◄────────────────────┘
                          │         │
                          └─────────┘
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (optional, but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/federated-clip-alignment.git
cd federated-clip-alignment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with default settings
python run_clip_alignment.py

# Run with a specific test domain
python run_clip_alignment.py --test-domain cartoon --lambda-align 0.1
```

### Advanced Options

```bash
# Enable differential privacy
python run_clip_alignment.py --use-dp-noise --dp-noise-std 0.01

# Customize embedding dimensions
python run_clip_alignment.py --small-dim 128 --mapper-hidden-dim 512

# Download PACS dataset if not exists
python run_clip_alignment.py --download-data
```

## Parameter Explanations

| Parameter | Description |
|-----------|-------------|
| `--test-domain` | Domain to use for testing (photo, art_painting, cartoon, sketch) |
| `--lambda-align` | Weight for alignment loss (higher = stronger CLIP alignment) |
| `--small-dim` | Dimension of small embedding (lower = more compression) |
| `--use-dp-noise` | Enable differential privacy noise |
| `--dp-noise-std` | Noise level for differential privacy (higher = more privacy) |

## Project Structure

```
federated-learning/
├── client/                    # Client-side code
│   ├── clip_alignment_client.py  # Client with CLIP alignment
│   └── trainer.py             # Base trainer
├── config/                    # Configuration files
│   └── config.yaml            # Main configuration
├── data/                      # Dataset handling
│   ├── download_pacs.py       # PACS dataset downloader
│   └── pacs_loader.py         # PACS data loader
├── model/                     # Model definitions
│   ├── clip_alignment.py      # CLIP alignment components
│   ├── cnn.py                 # CNN model with small embeddings
│   └── mapper.py              # Embedding mapper network
├── server/                    # Server-side code
│   ├── clip_coordinator.py    # Server with CLIP support
│   └── mapper_aggregator.py   # Aggregates mapper weights
├── utils/                     # Utilities
│   └── metrics.py             # Evaluation metrics
├── requirements.txt           # Dependencies
└── run_clip_alignment.py      # Main script
```

## Results and Evaluation

The system demonstrates several key advantages:

1. **Privacy Preservation**: Raw data never leaves the clients
2. **Semantic Knowledge Transfer**: Local models benefit from CLIP's knowledge
3. **Efficiency**: Compact embeddings (64D) aligned with rich CLIP space (512D)
4. **Cross-Domain Generalization**: Better transfer to new domains

## Innovative Aspects

1. **Bridging FL and Foundation Models**: Using CLIP as a local teacher in a federated architecture
2. **Compressed Embedding Mapping**: Mapping small embeddings to rich semantic space
3. **Privacy-Enhanced Aggregation**: Only sharing mapper weights
4. **Dual-Objective Training**: Balancing classification and alignment

## Limitations and Future Work

- Current implementation focuses on image classification
- Performance depends on domain similarity
- Adding more advanced privacy mechanisms
- Extending to other foundation models (DALL-E, GPT)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@software{jamali2023federated,
  author = {Jamali, Amirhossein},
  title = {Federated Learning with CLIP Alignment},
  year = {2023},
  url = {https://github.com/your-username/federated-clip-alignment}
}
```

## Contact

For questions or comments, please open an issue or contact the repository owner. 