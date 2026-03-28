# Data Preprocessing Pipeline

Comprehensive data preprocessing pipeline for image data preparation. Handles normalization, data type conversions, and augmentation for machine learning workflows.

## Features
- Data normalization (zero mean, unit variance)
- Image preprocessing for RGB data
- Data augmentation (flips, rotations)
- Label format conversion
- Type handling (PIL, numpy, tensor)

## Input Specifications
- Image format: RGB images (32x32) as uint8
- Value range: 0-255 (standard pixel values)
- Batch processing support
- Label format: Integer class indices

## Processing Pipeline
1. Load and validate input data formats
2. Apply normalization to image data
3. Convert to appropriate tensor types
4. Apply augmentation transformations
5. Format labels for model consumption

## Usage
```bash
python preprocess.py
```