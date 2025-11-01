import numpy as np
import myANFIS as anfis
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

#Load the dataset
data = np.genfromtxt('G:/My Drive/MSEE/Master_Thesis/Diabetes_Prediction/Datasets/CDC datasets/anfis_encod_train_dataset_150x5_v2.csv', delimiter=',', skip_header=1)  

# Define the batch size
batch_size = 50

# Settings for ANFIS model
epoch_n = 10
mf = 3
step_size = 0.1
decrease_rate = 0.9
increase_rate = 1.1

# Loop through the dataframe in batches of 150 rows
for i in range(0, len(data), batch_size):
    batch_data = data[i:i + batch_size]

    inputs = data[:, :-1]  # All columns except the last one are inputs
    output = data[:, -1:]  # The last column is the output
    ndata = data.shape[0]  # Data length

    # Fit the model for the current batch
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_input = scaler.fit_transform(inputs)
    
    # ANFIS train 
    bestnet, y_myanfis, RMSE = anfis.myanfis(data, scaled_input, epoch_n, mf, step_size, decrease_rate, increase_rate)


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

# Save the model
save_model(bestnet, 'trained_anfis_model.pkl')

bestnet=load_model('trained_anfis_model.pkl')
print("Training")
y_myanfis = anfis.evalmyanfis(bestnet, scaled_input)
anfis_predictions = y_myanfis

# For classification problem ( Round outputs to int)
anfis_predictions = np.round(anfis_predictions).astype(int)

print("anfis predictions")
print(anfis_predictions)

print("output")
print(output)

# Concatenate the output and anfis predictions
result = np.concatenate((output, anfis_predictions), axis=1)

# Save the result to a CSV file
np.savetxt('G:/My Drive/MSEE/Master_Thesis/Diabetes_Prediction/Datasets/CDC datasets/result_150r_10epoc_50batch.csv', result, delimiter=',', header='output,anfis_predictions', comments='')

#print("Result saved to 'result.csv'")

# Calculate the RMSE
rmse = anfis.calc_rmse(output,anfis_predictions)
msg = f'Total RMSE error myanfis: {rmse:.2f}'
print(msg)  # Print the message

"""
# Calculate the accuracy, precision, recall, f1, and RMSE1
accuracy = accuracy_score(output, anfis_predictions)
precision = precision_score(output, anfis_predictions)
recall = recall_score(output, anfis_predictions)
f1 = f1_score(output, anfis_predictions)
mse = mean_squared_error(output, anfis_predictions)
rmse1 = np.sqrt(mse)

# Create the results DataFrame
result = pd.DataFrame([[model_name, round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2), round(rmse1, 2)]],
                      columns=['Model', 'accuracy', 'precision', 'recall', 'f1', 'rmse1'])

print(result)

anfis.plot_Nodes(bestnet)

anfis.plot_mf(bestnet, data)

anfis.plot_predictions(output,anfis_predictions)

anfis.plot_r2(output,anfis_predictions)

print("Welcome !")

"""
