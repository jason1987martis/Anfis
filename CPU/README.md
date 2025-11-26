# CPU Version - ANFIS with Kalman Filter

This folder contains the CPU-based implementation of the Adaptive Neuro-Fuzzy Inference System (ANFIS) with **Kalman filter integration** for enhanced prediction accuracy.

## Overview

The CPU implementation provides a traditional NumPy-based ANFIS model with custom Gaussian bell-shaped membership functions. This version includes Kalman filtering capabilities embedded within the core algorithm for improved time-series forecasting.

## Features

- **Kalman Filter Integration**: Enhanced prediction accuracy for sequential data
- **Custom Membership Functions**: Gaussian bell-shaped functions for fuzzy logic
- **Divide and Conquer Approach**: Efficient data chunking for large datasets
- **Multiple Test Scenarios**: Various test files for different use cases
- **Educational Notebooks**: Detailed explanations of ANFIS functions and workflows

## Files Description

### Core Implementation
- `myANFIS.py` - Main ANFIS model with Kalman filter embedded
- `run_myanfis.py` - Script to run the ANFIS model
- `main.py` - Main execution script

### Test Files
- `test.py` - Basic ANFIS testing on Iris dataset
- `test2.py` - Advanced testing scenarios
- `test3.py` - Additional test configurations

### Data Processing
- `divide_data_in_chunks.py` - Utility to split large datasets into manageable chunks
- `getting_the_data.py` - Data loading and preprocessing utilities

### Documentation
- `ANFIS_functions_explanations.ipynb` - Detailed explanation of ANFIS functions
- `Test-py_File_explanation.ipynb` - Walkthrough of test.py file
- `Divide_and_Conquer_Code-Dr._Martis.txt` - Documentation on divide and conquer approach
- `readme(dataset).md` - Dataset information and structure

### Data Folders
- `input2/` - Input data directory
- `output/` - Model output and results
- `Results/` - Training results and metrics

## Installation

### Prerequisites

```bash
pip install numpy matplotlib scikit-fuzzy pandas
```

## Usage

### Basic Usage

1. **Run the basic test on Iris dataset:**

```bash
python test.py
```

This will load the iris.csv dataset, train the ANFIS model, and display the RMSE error.

2. **Run with custom dataset:**

```bash
python run_myanfis.py
```

3. **Execute main pipeline:**

```bash
python main.py
```

### Working with Large Datasets

For large datasets, use the divide and conquer approach:

```bash
python divide_data_in_chunks.py
```

This will split your data into chunks for more efficient processing.

### Advanced Testing

```bash
python test2.py  # Advanced scenarios
python test3.py  # Additional configurations
```

## Model Parameters

- **Number of Membership Functions**: Configurable per input variable
- **Learning Rate**: Adjustable for training convergence
- **Epochs**: Number of training iterations
- **Kalman Filter Parameters**: Q (process noise) and R (measurement noise)

## Output

The model generates:
- **Nodes**: ANFIS network nodes
- **Membership Functions**: Visualizations of fuzzy membership functions
- **Predictions**: Model predictions vs actual values
- **RMSE**: Root Mean Square Error metrics
- **Results**: Saved in the `Results/` and `output/` folders

## Example

```python
import numpy as np
from myANFIS import ANFIS

# Load your data
X_train = np.array([...])  # Input features
y_train = np.array([...])  # Target values

# Initialize ANFIS
anfis = ANFIS(n_inputs=4, n_rules=3, learning_rate=0.01)

# Train the model
anfis.train(X_train, y_train, epochs=100)

# Make predictions
predictions = anfis.predict(X_test)
```

## Notes

- The Kalman filter is embedded within `myANFIS.py` for real-time state estimation
- For time-series data, the Kalman filter significantly improves prediction accuracy
- Refer to the Jupyter notebooks for detailed explanations of each component
- The divide and conquer approach is recommended for datasets > 10,000 samples

## Performance

- **Training Speed**: Suitable for small to medium datasets (< 100K samples)
- **Memory Usage**: Low memory footprint
- **Accuracy**: Enhanced by Kalman filter for sequential data

## Troubleshooting

- **Convergence Issues**: Try adjusting learning rate or increasing epochs
- **Memory Errors**: Use divide_data_in_chunks.py for large datasets
- **Poor Accuracy**: Tune Kalman filter parameters (Q and R matrices)

## References

- Original ANFIS paper: Jang, J.S.R. (1993)
- Kalman Filter integration for ANFIS: Dr. Martis
- Source inspiration: https://github.com/namalhappy/anfis_from_scratch_python
