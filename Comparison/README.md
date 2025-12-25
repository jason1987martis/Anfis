# Comparison Tools

This directory contains scripts to compare the performance of GPU vs CPU training.

## Usage
To run the full pipeline (CPU run -> GPU run -> Comparison):

### Windows
```cmd
run_all.bat
```

This will:
1. Run `train_anfis.py` on CPU.
2. Run `train_anfis.py` on GPU.
3. Parse the logs and generate `comparison_report.html`.

## Manual Comparison
If you have already run the experiments, you can generate the report manually:

```bash
python compare.py
```

## Output
- `comparison_report.md`
- `comparison_report.html`
- `comparison_duration.png`
