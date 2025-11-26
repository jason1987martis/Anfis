# ANFIS Implementation - CPU & GPU Versions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains comprehensive implementations of Adaptive Neuro-Fuzzy Inference System (ANFIS) with both **CPU** (with Kalman filter) and **GPU** (PyTorch) versions, designed for different use cases and performance requirements.

## 🌟 Overview

ANFIS combines neural networks and fuzzy logic to create a powerful hybrid intelligent system. This repository provides:

- **CPU Version**: Traditional NumPy implementation with embedded Kalman filter for enhanced time-series prediction
- **GPU Version**: High-performance PyTorch implementation with CUDA acceleration for large-scale datasets
- **Comparison Tools**: Benchmarking utilities to evaluate performance differences

## 📁 Repository Structure

```
Anfis/
├── README.md                 # This file
├── CPU/                      # CPU implementation with Kalman filter
│   ├── README.md            # CPU version documentation
│   ├── myANFIS.py           # Core ANFIS with Kalman filter
│   ├── test.py, test2.py, test3.py
│   ├── Data processing scripts
│   └── Documentation notebooks
│
├── GPU/                      # GPU-accelerated PyTorch implementation
│   ├── README.md            # GPU version documentation
│   ├── model.py             # PyTorch ANFIS model
│   ├── trainer.py           # Training utilities
│   ├── Training scripts     # Various training modes
│   └── Explainability tools
│
└── Comparison/               # Performance benchmarking
    ├── README.md            # Comparison documentation
    └── Benchmark scripts    # CPU vs GPU comparison
```

## 🚀 Quick Start

### CPU Version (with Kalman Filter)

```bash
cd CPU
pip install numpy matplotlib scikit-fuzzy pandas
python test.py
```

### GPU Version (PyTorch)

```bash
cd GPU
pip install torch torchvision numpy pandas scikit-learn matplotlib
python train_anfis.py
```

### Benchmark Comparison

```bash
cd Comparison
python benchmark_cpu_epoch.py
```

## 📊 Feature Comparison

| Feature | CPU Version | GPU Version |
|---------|-------------|-------------|
| **Performance** | Good for small datasets | 10-50x faster for large datasets |
| **Special Features** | Kalman Filter embedded | Batch processing, mixed precision |
| **Dataset Size** | < 100K samples | Any size |
| **Memory Usage** | Low (0.5-2 GB RAM) | Higher (2-4 GB VRAM) |
| **Dependencies** | NumPy, scikit-fuzzy | PyTorch, CUDA |
| **Use Case** | Time-series, sequential data | Large-scale training, production |
| **Setup Difficulty** | Easy | Moderate (requires GPU) |
| **Inference Speed** | Fast | Very fast (batch) |

## 💡 Which Version Should I Use?

### Choose **CPU Version** if:
- Working with small to medium datasets (< 100K samples)
- Need Kalman filter for time-series forecasting
- No GPU available
- Simple setup preferred
- Sequential data processing required

### Choose **GPU Version** if:
- Large datasets (> 100K samples)
- Need fast training iterations
- GPU is available
- Production deployment with high throughput
- Running multiple experiments/hyperparameter tuning

## 📖 Documentation

Detailed documentation is available in each folder:

- **[CPU Documentation](CPU/README.md)** - CPU version usage, Kalman filter details, and examples
- **[GPU Documentation](GPU/README.md)** - GPU setup, training modes, and optimization tips
- **[Comparison Guide](Comparison/README.md)** - Benchmarking methodology and performance analysis

## 🔬 Key Features

### CPU Version Highlights
- ✅ Kalman filter integration for improved prediction accuracy
- ✅ Custom Gaussian membership functions
- ✅ Divide and conquer approach for large datasets
- ✅ Educational Jupyter notebooks with detailed explanations
- ✅ Multiple test scenarios and examples

### GPU Version Highlights
- ✅ CUDA acceleration for 10-50x speedup
- ✅ PyTorch automatic differentiation
- ✅ Multiple training modes (regression, binary, multi-class)
- ✅ Model explainability and interpretability tools
- ✅ Flexible architecture with configurable parameters
- ✅ Mixed precision training support

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version
```

### CPU Version Dependencies

```bash
pip install numpy matplotlib scikit-fuzzy pandas
```

### GPU Version Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pandas scikit-learn matplotlib

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📚 Example Usage

### CPU Example

```python
import numpy as np
from myANFIS import ANFIS

# Load data
X_train = np.array([...])  # Your input features
y_train = np.array([...])  # Your target values

# Initialize and train
anfis = ANFIS(n_inputs=4, n_rules=3, learning_rate=0.01)
anfis.train(X_train, y_train, epochs=100)

# Predict
predictions = anfis.predict(X_test)
```

### GPU Example

```python
import torch
from model import ANFIS

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = ANFIS(n_inputs=4, n_rules=16, n_outputs=1).to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```

## 🎯 Use Cases

- **Time-Series Forecasting** (CPU with Kalman filter)
- **Large-Scale Classification** (GPU)
- **Regression Tasks** (Both versions)
- **Real-Time Prediction** (CPU for low latency)
- **Batch Processing** (GPU for high throughput)
- **Research and Experimentation** (Both versions)

## 📈 Performance Benchmarks

**Training Speed Comparison (10K samples, 100 epochs):**
- CPU: ~245 seconds
- GPU: ~8.7 seconds
- **Speedup: 28x**

See [Comparison/README.md](Comparison/README.md) for detailed benchmarks.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source. See LICENSE file for details.

## 🙏 Acknowledgments

- Original ANFIS concept: Jang, J.S.R. (1993)
- Kalman filter integration: Dr. Martis
- Base implementation inspiration: [namalhappy/anfis_from_scratch_python](https://github.com/namalhappy/anfis_from_scratch_python)

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the documentation in each folder's README

## 🔗 Useful Links

- [CPU Version Documentation](CPU/README.md)
- [GPU Version Documentation](GPU/README.md)
- [Performance Comparison](Comparison/README.md)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ANFIS Theory](https://en.wikipedia.org/wiki/Adaptive_neuro_fuzzy_inference_system)

---

**Happy coding! 🚀**
