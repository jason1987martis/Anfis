"""Example script demonstrating how to train and use the ANFIS model."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import ANFISNetwork

def prepare_data(filepath='D:/code_space/Anfis/input/binary_no-head_BMI-AGE-Income-PhysHlth_200x5.csv', test_size=0.2, batch_size=32):
    """Load and prepare the data for training."""
    # Load the data (no header)
    df = pd.read_csv(filepath, header=None)
    
    # Remove any rows with NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # First 4 columns are features (BMI, AGE, Income, PhysHlth)
    X = df.iloc[:, :4].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    
    # Scale the features to [0, 1] range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale target to [0, 1] range using MinMaxScaler
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled.reshape(-1, 1))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, scaler

def train_model(model, train_loader, test_loader, epochs=100, learning_rate=0.0001):
    """Train the ANFIS model."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_val, y_val in test_loader:
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item()
                val_batches += 1
            
            val_loss /= val_batches
            
            # Debug information
            max_val = max(p.abs().max() for p in model.parameters())
            
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_anfis_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Max param: {max_val:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Load the best model
    model.load_state_dict(torch.load('best_anfis_model.pth'))
    return model

def evaluate_model(model, test_loader):
    """Evaluate the trained model."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            outputs = model(X_test)
            loss = criterion(outputs, y_test)
            total_loss += loss.item()
            batch_count += 1
    
    avg_test_loss = total_loss / batch_count
    print(f'\nTest Loss: {avg_test_loss:.4f}')
    return avg_test_loss

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load and prepare data
    train_loader, test_loader, scaler = prepare_data()
    
    # Create ANFIS model
    input_dim = 4  # Number of features (BMI, AGE, Income, PhysHlth)
    mf_count = 2  # Number of membership functions per input
    model = ANFISNetwork(input_dim=input_dim, mf_count=mf_count)
    
    # Get a sample batch for initialization
    X_sample = next(iter(train_loader))[0]
    
    # Initialize membership functions based on data
    model.initialize_memberships(X_sample)
    
    # Train the model
    model = train_model(model, train_loader, test_loader, epochs=100)
    
    # Evaluate the model
    evaluate_model(model, test_loader)
    
    print("\nBest model saved to 'best_anfis_model.pth'")

if __name__ == "__main__":
    main()