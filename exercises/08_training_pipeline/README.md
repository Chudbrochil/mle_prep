# Training Pipeline

End-to-end training pipeline for neural network models with comprehensive metrics tracking, checkpointing, and visualization.

## Features
- Multi-layer perceptron (MLP) architecture
- Training and validation loops
- Accuracy and loss tracking
- Model checkpointing
- Training progress visualization
- Memory management during training

## Model Architecture
- Input: 784 features (flattened images)
- Hidden layers: 256 → 128 units with ReLU activation
- Output: 10 classes for classification
- Dropout regularization for overfitting prevention

## Training Components
1. **Data Loading**: TensorDataset and DataLoader integration
2. **Training Loop**: Epoch-based training with batch processing
3. **Validation**: Model evaluation on validation set
4. **Metrics**: Loss and accuracy tracking
5. **Checkpointing**: Model state saving at regular intervals
6. **Visualization**: Training progress plots

## Expected Output
When functioning correctly:
- Trains model for specified epochs
- Reports training and validation metrics
- Saves model checkpoints
- Generates training progress plots
- Maintains stable memory usage

## Usage
```bash
python training_loop.py
```