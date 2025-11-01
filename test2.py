import numpy as np
import myANFIS as anfis
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

data = np.genfromtxt('G:/My Drive/MSEE/Master_Thesis/Diabetes_Prediction/Jupyter Notebooks/ANFIS Implementation/anfis_code/input/binary_no-head_BMI-AGE-Income-PhysHlth_300x5.csv', delimiter=',', skip_header=1)  
# Divide data into input and output
inputs = data[:, :-1]  # All columns except the last one are inputs
output = data[:, -1:]  # The last column is the output
ndata = data.shape[0]  # Data length

# Check for NaN values in the data
nan_count = np.isnan(inputs).sum()
print(f"Number of NaN values: {nan_count}")
#inputs[np.isnan(inputs)] = 40

# Use the function to save the 'bestnet' model
def save_model(model, filename):
    """
    Saves the trained model to a file.

    Args:
        model (object): The trained model to be saved.
        filename (str): The name of the file where the model will be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully to '{filename}'")

# Use the function to save the 'bestnet' model
def load_model(filename):
    """
    Loads a trained model from a file.

    Args:
        filename (str): The name of the file from which the model will be loaded.

    Returns:
        object: The loaded model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from '{filename}'")
    return model

#-------------------------

# Load the CSV file into a DataFrame
#file_path = 'G:/My Drive/MSEE/Master_Thesis/Diabetes_Prediction/Jupyter Notebooks/ANFIS Implementation/anfis_code/input/binary_no-head_BMI-AGE-Income-PhysHlth_300x5.csv'
#data = pd.read_csv(file_path, header=None)  # Assuming there is no header in the file

# Splitting the DataFrame into input and output
#input_columns = data.iloc[:, :-1]  # Select all columns except the last one
#output_column = data.iloc[:, -1]   # Select only the last column

# Naming the tables
#inputs = input_columns
#output = output_column

#--------------------------

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_input = scaler.fit_transform(inputs)


# Settings for ANFIS model
epoch_n = 1
mf = 3
step_size = 0.1
decrease_rate = 0.9
increase_rate = 1.1

# ANFIS train 
bestnet, y_myanfis, RMSE = anfis.myanfis(data, scaled_input, epoch_n, mf, step_size, decrease_rate, increase_rate)
save_model(bestnet, 'trained_anfis_model.pkl')
bestnet=load_model('trained_anfis_model.pkl')
print("Training")
y_myanfis = anfis.evalmyanfis(bestnet, scaled_input)

anfis_predictions = y_myanfis

# For classification problem ( Round outputs to int)
anfis_predictions = np.round(anfis_predictions).astype(int)

# Calculate the RMSE
rmse = anfis.calc_rmse(output,anfis_predictions)

msg = f'Total RMSE error myanfis: {rmse:.2f}'
print(msg)  # Print the message

anfis.plot_Nodes(bestnet)

anfis.plot_mf(bestnet, data)

anfis.plot_predictions(output,anfis_predictions)

anfis.plot_r2(output,anfis_predictions)

print("Welcome !")
