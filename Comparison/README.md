# CPU vs GPU Performance Comparison

This folder contains benchmark scripts and tools to compare the performance of CPU and GPU implementations of ANFIS.

## Overview

Compare training time, memory usage, and accuracy between the CPU (with Kalman filter) and GPU (PyTorch) versions of ANFIS to help you choose the right implementation for your use case.

## Files Description

- `benchmark_cpu_epoch.py` - Benchmark CPU performance per epoch
- `train_on_input2_cpu.py` - CPU training script for input2 dataset
- `train_on_input2.py` - GPU training script for input2 dataset (copy from GPU folder)

## Quick Comparison

| Feature | CPU Version | GPU Version |
|---------|-------------|-------------|
| **Speed** | Slower for large datasets | 10-50x faster |
| **Memory** | Lower RAM usage | Higher VRAM usage |
| **Dataset Size** | < 100K samples | Any size |
| **Special Feature** | Kalman Filter | Batch processing |
| **Dependencies** | NumPy, scikit-fuzzy | PyTorch, CUDA |
| **Ease of Use** | Simple setup | Requires GPU setup |

## Running Benchmarks

### Benchmark CPU Performance

```bash
python benchmark_cpu_epoch.py
```

This will:
- Measure time per epoch
- Track memory usage
- Display training metrics
- Save results to file

### Compare on Same Dataset

1. **Run CPU version:**

```bash
python train_on_input2_cpu.py
```

2. **Run GPU version:**

```bash
python train_on_input2.py
```

3. **Compare results:**
   - Training time
   - Final RMSE/accuracy
   - Memory usage
   - Model size

## Benchmark Metrics

### Performance Metrics
- **Training Time**: Total time to train for N epochs
- **Time per Epoch**: Average time per epoch
- **Inference Time**: Time to make predictions
- **Throughput**: Samples processed per second

### Resource Metrics
- **CPU Usage**: Percentage of CPU utilization
- **Memory Usage**: RAM consumption (CPU) or VRAM (GPU)
- **Power Consumption**: Watts used during training

### Model Metrics
- **RMSE**: Root Mean Square Error
- **Accuracy**: Classification accuracy (if applicable)
- **Convergence Speed**: Epochs to reach target loss

## Example Output

```
=== CPU Version ===
Training Time: 245.3 seconds
Time per Epoch: 2.45 seconds
Memory Usage: 1.2 GB RAM
Final RMSE: 0.0234
Converged at epoch: 87

=== GPU Version ===
Training Time: 8.7 seconds (28x faster)
Time per Epoch: 0.087 seconds
Memory Usage: 2.1 GB VRAM
Final RMSE: 0.0231
Converged at epoch: 85

=== Speedup ===
GPU is 28.2x faster than CPU
```

## Custom Benchmarks

Create your own benchmark:

```python
import time
import psutil
import torch

# CPU benchmark
start_time = time.time()
process = psutil.Process()

# Your CPU training code here
# ...

cpu_time = time.time() - start_time
cpu_memory = process.memory_info().rss / 1024 ** 3  # GB

print(f"CPU Time: {cpu_time:.2f}s")
print(f"CPU Memory: {cpu_memory:.2f} GB")

# GPU benchmark
if torch.cuda.is_available():
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Your GPU training code here
    # ...
    
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
    
    print(f"GPU Time: {gpu_time:.2f}s")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")
```

## When to Use Each Version

### Use CPU Version When:
- Dataset is small (< 10K samples)
- You need Kalman filter for time-series
- No GPU available
- Quick prototyping
- Lower memory systems
- Sequential data processing required

### Use GPU Version When:
- Large datasets (> 100K samples)
- Need fast training
- GPU is available
- Batch processing preferred
- Production deployment with high throughput
- Multiple experiments/hyperparameter tuning

## Detailed Comparison Analysis

### Training Speed

**Small Dataset (< 10K samples):**
- CPU: ~2-5 seconds per epoch
- GPU: ~0.1-0.5 seconds per epoch
- Speedup: 5-10x

**Medium Dataset (10K-100K samples):**
- CPU: ~10-60 seconds per epoch
- GPU: ~0.5-2 seconds per epoch
- Speedup: 20-30x

**Large Dataset (> 100K samples):**
- CPU: ~60-300+ seconds per epoch
- GPU: ~2-10 seconds per epoch
- Speedup: 30-50x

### Memory Usage

- **CPU**: 0.5-2 GB RAM (depends on dataset size)
- **GPU**: 1-4 GB VRAM (includes model + batch data)

### Accuracy Comparison

- Both versions achieve similar final accuracy
- CPU with Kalman filter may perform better on sequential data
- GPU version converges faster (fewer epochs needed)

## Profiling Tools

### CPU Profiling

```bash
python -m cProfile -o cpu_profile.prof train_on_input2_cpu.py
python -m pstats cpu_profile.prof
```

### GPU Profiling

```bash
python -m torch.utils.bottleneck train_on_input2.py
```

Or use NVIDIA tools:

```bash
nvprof python train_on_input2.py
```

## Tips for Fair Comparison

1. **Same Dataset**: Use identical train/test splits
2. **Same Hyperparameters**: Keep learning rate, epochs consistent
3. **Same Architecture**: Use same number of rules and membership functions
4. **Multiple Runs**: Average results over 5+ runs
5. **Warm-up**: Run once to warm up GPU before benchmarking
6. **Consistent Environment**: Close other applications, consistent power settings

## Visualization

Generate comparison charts:

```python
import matplotlib.pyplot as plt

# Training time comparison
plt.figure(figsize=(10, 6))
plt.bar(['CPU', 'GPU'], [cpu_time, gpu_time])
plt.ylabel('Training Time (seconds)')
plt.title('CPU vs GPU Training Time')
plt.savefig('training_time_comparison.png')
```

## Contributing

To add new benchmarks:
1. Create a new benchmark script
2. Follow the naming convention: `benchmark_<feature>.py`
3. Document metrics clearly
4. Update this README

## Notes

- GPU performance depends on GPU model (RTX 3090 > GTX 1660)
- CPU performance depends on number of cores and RAM
- Results may vary based on hardware configuration
- Power consumption: GPU typically uses more power but faster overall
- For production: Consider GPU for training, CPU for inference

## System Requirements for Benchmarking

### Minimum
- CPU: 4 cores, 8GB RAM
- GPU: NVIDIA GPU with 4GB VRAM (optional)

### Recommended
- CPU: 8+ cores, 16GB RAM
- GPU: NVIDIA RTX series with 8GB+ VRAM
