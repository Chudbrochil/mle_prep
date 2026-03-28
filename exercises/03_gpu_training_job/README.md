# GPU Training Job

A PyTorch training script designed to run efficiently on GPU hardware. The script includes a CNN model for MNIST-like image classification with support for both CPU and GPU execution.

## Model Architecture
- Convolutional layers with ReLU activation
- Max pooling for spatial reduction
- Fully connected layers for classification
- Dropout for regularization

## Features
- Automatic device detection (CUDA/CPU)
- Memory usage monitoring
- Training and evaluation loops
- Adam optimizer with cross-entropy loss

## Expected Output
When running correctly:
1. Detects and reports available compute device
2. Creates and trains the model on dummy data
3. Reports training and validation metrics
4. Displays memory usage information
5. Completes training successfully

## Usage
```bash
python train.py
```