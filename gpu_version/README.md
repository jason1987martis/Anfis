# GPU Version

This directory contains the GPU-optimized training implementation for ANFIS.

## Setup
1. Ensure you have a CUDA-capable GPU and drivers installed.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: Ensure you install the CUDA version of PyTorch).

## Usage
Run the training wrapper using the provided batch file or Python script.

### Windows (Batch)
```cmd
run_gpu.bat --script train_anfis.py --seed 42
```

### Python
```bash
python run_gpu.py --script train_anfis.py --seed 42
```

## Configuration
Edit `run_config.yml` to change default hyperparameters, though note that some scripts may have hardcoded values.

## Output
- Logs are saved to `logs/`.
- Models are saved to `models/` (if the script supports it) or `output/` (original script location).
