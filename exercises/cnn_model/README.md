# CNN Model Architecture

Deep convolutional neural network for image classification tasks. Supports variable input sizes and includes comprehensive architecture testing.

## Architecture Details
- **Conv Layer 1**: 3→32 channels, 5x5 kernel with padding
- **Conv Layer 2**: 32→64 channels, 3x3 kernel  
- **Conv Layer 3**: 64→128 channels, 3x3 kernel
- **Pooling**: 2x2 max pooling between conv layers
- **FC Layers**: Adaptive fully connected layers based on input size
- **Output**: Configurable number of classes

## Testing Suite
The script includes comprehensive testing for:
- Multiple input image sizes (32x32, 64x64)
- Different batch sizes
- Forward and backward pass validation
- Training step simulation
- Architecture compatibility checks

## Expected Functionality
When running correctly:
1. Build CNN model with proper layer dimensions
2. Handle variable input sizes automatically
3. Perform forward passes for different image sizes
4. Complete training steps with loss calculation
5. Report successful architecture validation

## Usage
```bash
python model.py
```