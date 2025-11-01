import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'G:/My Drive/MSEE/Master_Thesis/Diabetes_Prediction/Jupyter Notebooks/ANFIS Implementation/anfis_code/input/binary_no-head_BMI-AGE-Income-PhysHlth_300x5.csv'
data = pd.read_csv(file_path, header=None)  # Assuming there is no header in the file

# Splitting the DataFrame into input and output
input_columns = data.iloc[:, :-1]  # Select all columns except the last one
output_column = data.iloc[:, -1]   # Select only the last column

# Naming the tables
input = input_columns
output = output_column

# Displaying the first few rows to verify
print("Input Table:")
print(input.head())
print("\nOutput Table:")
print(output.head())