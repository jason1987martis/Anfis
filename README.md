# ANFIS Implementation - CPU & GPU Versions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains comprehensive implementations of Adaptive Neuro-Fuzzy Inference System (ANFIS) with both **CPU** (with Kalman filter) and **GPU** (PyTorch) versions, designed for different use cases and performance requirements.

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



## 🔗 Useful Links

- [CPU Version Documentation](cpu_version/README.md)
- [GPU Version Documentation](gpu_version/README.md)
- [Performance Comparison](comparison/README.md)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ANFIS Theory](https://en.wikipedia.org/wiki/Adaptive_neuro_fuzzy_inference_system)


