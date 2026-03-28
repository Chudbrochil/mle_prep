# Image Classification Model

A PyTorch-based image classification model for processing 28x28 grayscale images. The model uses a simple neural network architecture with fully connected layers.

## Architecture
- Input: Flattened 28x28 images (784 features)
- Hidden layer: 128 units with ReLU activation
- Output: 10 classes for classification

## Expected Functionality
The model should:
1. Process batch data of 28x28 images
2. Flatten images for fully connected layers
3. Perform forward pass through the network
4. Calculate loss and display training metrics
5. Handle matrix operations and tensor transformations

## Usage
```bash
python train_model.py
```

## Files
- `train_model.py` - Main training script with model definition