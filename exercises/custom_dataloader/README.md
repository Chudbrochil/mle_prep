# Custom Dataset Implementation

A PyTorch custom Dataset class for loading synthetic image data. Includes DataLoader integration for batch processing and data transformations.

## Components
- `ImageDataset`: Custom dataset class for synthetic image data
- `TransformDataset`: Wrapper for applying transformations
- `collate_fn`: Custom batch collation function
- Data augmentation pipeline

## Dataset Specifications
- Image size: 32x32 RGB (3 channels)
- Number of classes: 10
- Data format: Synthetic random data for testing
- Batch processing support

## Expected Functionality
The script should:
1. Create a custom dataset with synthetic image data
2. Implement proper Dataset interface (__getitem__, __len__)
3. Apply transformations to the data
4. Create DataLoader for batch iteration
5. Successfully iterate through batches

## Usage
```bash
python dataset.py
```