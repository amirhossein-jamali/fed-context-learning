# Federated Learning with CLIP Alignment

This project implements a federated learning system with CLIP alignment for domain generalization on the PACS dataset.

## Overview

The system implements federated learning where each client:

1. Extracts a compact local embedding vector (`z_small ∈ ℝ⁶⁴`) from a CNN model
2. Uses a local CLIP model to compute semantic embeddings (`z_clip ∈ ℝ⁵¹²`) for its images
3. Trains a mapping function (`φ: ℝ⁶⁴ → ℝ⁵¹²`) to align the small embeddings to the CLIP space
4. Only shares the weights of `φ` with the server (preserving data privacy)

## Key Components

- **CNN Model**: Outputs both classification logits and compact embeddings (`z_small`)
- **CLIP Model**: Used as a frozen teacher to provide semantic embeddings
- **Mapper Network (φ)**: MLP that maps `z_small` to the CLIP embedding space
- **Combined Loss**: Classification loss + alignment loss
- **Differential Privacy**: Optional Gaussian noise applied to small embeddings for enhanced privacy

## Usage

Run the training with:

```bash
python run_clip_alignment.py --test-domain cartoon --lambda-align 0.1 --small-dim 64
```

### Command-line Arguments

- `--config`: Path to config file (default: config/config.yaml)
- `--seed`: Random seed (default: 42)
- `--test-domain`: Domain to use for testing
- `--device`: Device to use (cuda or cpu)
- `--download-data`: Download PACS dataset if not exists
- `--save-client-models`: Save individual client models for each domain
- `--lambda-align`: Weight for alignment loss (default: 0.1)
- `--use-dp-noise`: Apply differential privacy noise
- `--dp-noise-std`: Standard deviation for DP noise (default: 0.01)
- `--small-dim`: Dimension of small embedding (default: 64)
- `--mapper-hidden-dim`: Hidden dimension of mapper network (default: 256)

## Algorithm Details

1. Each client trains a CNN model that outputs both classification logits and a small embedding (`z_small`)
2. A frozen CLIP model is used to compute target embeddings (`z_clip`) locally on each client
3. A mapper network (`φ`) is trained to map `z_small` to `z_clip`
4. The mapper weights are aggregated centrally using FedAvg
5. After training, the system can leverage both the task-specific `z_small` and the semantic-aligned `φ(z_small)`

## Privacy Considerations

- Raw data never leaves the client
- CLIP embeddings are computed locally and never shared
- Only the mapper weights are transmitted to the server
- Optional differential privacy noise can be added to `z_small` before mapping

## Implementation Notes

- The CNN model is based on a simple convolutional architecture
- CLIP uses the ViT-B/32 architecture (512-dimensional embeddings)
- The mapper is a two-layer MLP with configurable hidden dimensions
- FedAvg is used for both model and mapper aggregation

## Output and Visualization

The system generates several metrics and visualizations:

- Training and testing accuracy over communication rounds
- Classification and alignment losses
- Client-specific performance metrics
- CLIP alignment loss convergence plots

## Directory Structure

- `model/cnn.py`: CNN model definition with small embedding output
- `model/mapper.py`: Embedding mapper (φ) implementation
- `model/clip_alignment.py`: CLIP model and alignment loss functions
- `client/clip_alignment_client.py`: Client implementation with CLIP alignment
- `server/mapper_aggregator.py`: Server-side mapper aggregation
- `server/clip_coordinator.py`: Federated learning coordinator with CLIP alignment
- `run_clip_alignment.py`: Main script to run the system

## Requirements

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- CLIP (OpenAI)
- matplotlib
- numpy
- pillow
- ftfy (for CLIP)
- regex (for CLIP) 