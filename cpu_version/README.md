# CPU Version

This directory contains the CPU-only training implementation for ANFIS. It enforces CPU execution even if a GPU is present.

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the training wrapper using the provided batch file or Python script.

### Windows (Batch)
```cmd
run_cpu.bat --script train_anfis.py --seed 42
```

### Python
```bash
python run_cpu.py --script train_anfis.py --seed 42
```

## Configuration
Edit `run_config.yml` to change default hyperparameters.

## Output
- Logs are saved to `logs/`.
- Models are saved to `models/`.
