# Amazon ML Challenge - Entity Value Extraction

A deep learning solution for extracting entity values from product images and text using a custom Resformer (ResNet + Transformer) architecture. This project tackles the challenge of automatically identifying and extracting specific product attributes like dimensions, weight, voltage, and volume from e-commerce product images.

## Problem Statement

The Amazon ML Challenge focuses on extracting entity values from product images where the entity name is provided. For example, given an image of a product and the entity "item_weight", the model should predict the weight value with its appropriate unit (e.g., "2.5 kilogram").

## Architecture Overview

### Resformer Model
The solution implements a novel **Resformer** architecture that combines:

- **ResNet Encoder**: Custom ResNet with self-attention mechanisms for robust image feature extraction
- **Transformer Encoder-Decoder**: Processes image patches and entity types to generate structured text outputs
- **Multi-modal Fusion**: Combines visual features with entity type embeddings

### Key Components

1. **ResNet with Self-Attention**: Enhanced ResNet blocks with integrated self-attention for better spatial feature understanding
2. **Patch Embedding**: Converts processed image features into patch-based embeddings for transformer input
3. **Entity Type Embedding**: Encodes the target entity type (weight, volume, voltage, etc.) as contextual information
4. **Transformer Decoder**: Generates token sequences representing the entity value and unit

## Project Structure

```
├── amazon files/
│   ├── code/
│   │   ├── constants.py      # Entity-unit mappings and validation
│   │   ├── utils.py          # Image download and text parsing utilities
│   │   ├── sanity.py         # Output validation and format checking
│   │   └── test.ipynb        # Basic testing and validation notebook
│   └── data/                 # Dataset CSV files
├── dataset.py                # Custom PyTorch dataset implementation
├── models.py                 # Resformer architecture implementation
├── tokens.py                 # Input/output tokenization mappings
├── trainer.py                # Training loop and utilities
├── training_models.ipynb     # Main training notebook
└── download_data.ipynb       # Data preparation and image downloading
```

## Technical Implementation

### Data Processing
- **Image Handling**: Automatic download with retry mechanisms and placeholder generation for failed downloads
- **Text Parsing**: Robust parsing of entity values with unit validation and common mistake correction
- **Tokenization**: Custom tokenization scheme for entity types and output values

### Model Architecture Details
- **Input Resolution**: 400x400 pixel images with ResNet preprocessing
- **Embedding Dimension**: 2048-dimensional feature representations
- **Patch Size**: 16x16 patches for transformer processing
- **Sequence Length**: Maximum 64 tokens for output generation
- **Multi-head Attention**: 16 attention heads in transformer layers

### Training Features
- **Mixed Precision Training**: CUDA AMP integration for efficient GPU utilization
- **Gradient Scaling**: Automatic loss scaling for numerical stability
- **Progressive Saving**: Model checkpoints every 100 batches
- **Real-time Monitoring**: Live prediction sampling during training

## Usage

### Environment Setup
```bash
pip install torch torchvision pandas pillow tqdm numpy pathlib
```

### Data Preparation
```python
from utils import download_images
download_images(train['image_link'], 'dataset/train', allow_multiprocessing=True)
```

### Training
```python
from models import ResformerEncoder, ResformerDecoder
from trainer import model_trainer

encoder = ResformerEncoder().to(device)
decoder = ResformerDecoder().to(device)
model_trainer(encoder, decoder, dataloader, loss_fn, optimizer, scaler, device, epochs)
```

### Validation
```bash
python sanity.py --test_filename sample_test.csv --output_filename predictions.csv
```

## Entity Types Supported

The model handles 8 different entity types:
- **Dimensions**: width, depth, height (cm, ft, in, m, mm, yd)
- **Weight**: item_weight, maximum_weight_recommendation (g, kg, μg, mg, oz, lb, ton)
- **Electrical**: voltage (kV, mV, V), wattage (kW, W)
- **Volume**: item_volume (cl, ft³, in³, cup, dl, fl oz, gal, L, etc.)

## Key Features

- **Multi-modal Learning**: Combines visual and textual information effectively
- **Robust Parsing**: Handles various text formats and common unit spelling variations
- **Scalable Architecture**: Efficient training with gradient scaling and mixed precision
- **Comprehensive Validation**: Built-in sanity checking for output format compliance
- **Production Ready**: Includes error handling, logging, and model checkpointing

## Results and Performance

The Resformer architecture demonstrates strong performance in extracting entity values from product images by leveraging both visual context and entity type information. The self-attention mechanisms in the ResNet backbone help focus on relevant image regions, while the transformer decoder generates accurate value-unit pairs.

This solution provides a robust foundation for automated product attribute extraction in e-commerce applications, with potential extensions to additional entity types and multi-language support.