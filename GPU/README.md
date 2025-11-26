# GPU Version - PyTorch ANFIS

This folder contains the GPU-accelerated implementation of ANFIS using PyTorch, enabling fast training on large datasets with CUDA support.

## Overview

The GPU version leverages PyTorch's automatic differentiation and GPU acceleration to train ANFIS models efficiently. This implementation is optimized for large-scale datasets and supports various training modes including regression, binary classification, and multi-class classification.

## Features

- **GPU Acceleration**: CUDA-enabled for fast training
- **PyTorch Backend**: Automatic differentiation and optimized operations
- **Multiple Training Modes**: Regression, binary, and multi-class classification
- **Batch Processing**: Efficient mini-batch training
- **Model Explainability**: Tools to interpret ANFIS decisions
- **Flexible Architecture**: Configurable membership functions and rules

## Files Description

### Core Implementation
- `model.py` - PyTorch ANFIS model architecture
- `trainer.py` - Training loop and optimization logic
- `dataset_loader.py` - Data loading and preprocessing utilities

### Training Scripts
- `train_anfis.py` - General ANFIS training script
- `train_anfis_binary.py` - Binary classification training
- `train_anfis_regression.py` - Regression task training
- `train_on_input2.py` - Training on input2 dataset (GPU version)
- `test_brfss.py` - Testing on BRFSS dataset

### Utilities
- `explainability.py` - Model interpretation and visualization tools
- `explainability.md` - Documentation on explainability features
- `parameter_fitting_options.md` - Guide to parameter tuning
- `__init__.py` - Package initialization

### Model Files
- `anfis_model.pth` - Saved trained model weights

## Installation

### Prerequisites

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

### CUDA Setup (for GPU acceleration)

Ensure you have CUDA-compatible GPU and drivers installed:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Basic Training (Regression)

```bash
python train_anfis_regression.py
```

This will:
1. Load your dataset
2. Initialize ANFIS model on GPU (if available)
3. Train using Adam optimizer
4. Save model to `anfis_model.pth`
5. Display training metrics

### Binary Classification

```bash
python train_anfis_binary.py
```

### Custom Dataset Training

For the input2 dataset:

```bash
python train_on_input2.py
```

### Testing on BRFSS Dataset

```bash
python test_brfss.py
```

## Configuration

### Model Parameters

Edit the training scripts to configure:

```python
# Model architecture
n_inputs = 4          # Number of input features
n_rules = 16          # Number of fuzzy rules
n_outputs = 1         # Number of outputs

# Training parameters
batch_size = 32       # Mini-batch size
learning_rate = 0.001 # Learning rate
epochs = 100          # Number of epochs

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Membership Functions

Configure in `model.py`:
- Gaussian
- Bell-shaped
- Triangular
- Trapezoidal

## Model Explainability

Generate explanations for model predictions:

```bash
python explainability.py
```

This will:
- Visualize membership functions
- Show rule activations
- Display feature importance
- Generate prediction heatmaps

See `explainability.md` for detailed documentation.

## Example Code

```python
import torch
from model import ANFIS
from dataset_loader import load_dataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
X_train, y_train, X_test, y_test = load_dataset('your_data.csv')

# Initialize model
model = ANFIS(n_inputs=4, n_rules=16, n_outputs=1).to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), 'anfis_model.pth')

# Predict
model.eval()
with torch.no_grad():
    predictions = model(X_test)
```

## Advanced Features

### Parameter Fitting Options

Refer to `parameter_fitting_options.md` for:
- Hybrid learning (LSE + Gradient Descent)
- Backpropagation only
- Custom optimization strategies

### Model Saving and Loading

```python
# Save
torch.save(model.state_dict(), 'anfis_model.pth')

# Load
model = ANFIS(n_inputs=4, n_rules=16, n_outputs=1)
model.load_state_dict(torch.load('anfis_model.pth'))
model.eval()
```

## Performance Tips

1. **Batch Size**: Increase for better GPU utilization (try 64, 128, 256)
2. **Number of Rules**: More rules = more capacity but slower training
3. **Learning Rate**: Start with 0.001, adjust based on convergence
4. **Data Normalization**: Always normalize inputs for better convergence
5. **Mixed Precision**: Use `torch.cuda.amp` for faster training on modern GPUs

## GPU Memory Management

```python
# Clear cache if running out of memory
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
torch.utils.checkpoint.checkpoint(model, inputs)
```

## Benchmarking

See the `../Comparison/` folder for CPU vs GPU performance benchmarks.

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or number of rules
- **Slow Training**: Ensure CUDA is available and being used
- **NaN Loss**: Reduce learning rate or check data normalization
- **Poor Convergence**: Increase epochs or adjust learning rate

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **VRAM**: 4GB+ for large models
- **Python**: 3.8+
- **PyTorch**: 1.10+

## Notes

- GPU version is 10-50x faster than CPU for large datasets
- Supports multi-GPU training (modify trainer.py for DataParallel)
- Compatible with TensorBoard for training visualization
- Models trained on GPU can be loaded on CPU and vice versa
