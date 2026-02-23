import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from myANFIS import myanfis, plot_Nodes, plot_mf, plot_predictions, plot_r2, calc_rmse, calc_r2

# Load the data
data = np.loadtxt('input/iris.csv', delimiter=',')

# Parameters for ANFIS
epoch_n = 10  # Number of epochs
mf = 2        # Number of membership functions
step_size = 0.1  # Learning rate
decrease_rate = 0.9  # Rate at which step size decreases
increase_rate = 1.1  # Rate at which step size increases

# Prepare input and output data
inputs = data[:, :-1]  # All columns except the last one
output = data[:, -1:]  # Last column

# Train ANFIS
print("Training ANFIS...")
bestnet, anfis_predictions, RMSE = myanfis(data, inputs, epoch_n, mf, step_size, decrease_rate, increase_rate)

# Plot results
print("\nPlotting results...")
plot_Nodes(bestnet)
plot_mf(bestnet, data)
plot_predictions(output, anfis_predictions)
plot_r2(output, anfis_predictions)

# Calculate performance metrics
rmse = calc_rmse(output, anfis_predictions)
r2 = calc_r2(output, anfis_predictions)

print(f"\nFinal Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")