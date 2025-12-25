import numpy as np

def read_and_chunk_data(file_path, chunk_size):
    # Read the data from a text file
    data = np.genfromtxt(file_path, delimiter=',')  # Update delimiter if different
    
    # Calculate the number of chunks
    num_chunks = len(data) // chunk_size + (len(data) % chunk_size > 0)
    
    # Divide the data into chunks and save each chunk to a separate file
    for i in range(num_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        # Save each chunk to a separate file
        np.savetxt(f'chunk_{i + 1}.csv', chunk, delimiter=',', fmt='%s')  # Update fmt based on your data type

# Example usage
file_path = 'your_data_file.csv'  # The path to your input file
chunk_size = 100  # Number of rows per chunk
read_and_chunk_data(file_path, chunk_size)