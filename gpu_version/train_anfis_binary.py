"""Example script demonstrating how to train and use the ANFIS model."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import ANFISNetwork

def prepare_data(filepath='D:/code_space/Anfis/input/binary_no-head_BMI-AGE-Income-PhysHlth_200x5.csv', test_size=0.2, batch_size=16):
    """Load and prepare the data for training."""
    # Load the data (no header)
    df = pd.read_csv(filepath, header=None)
    
    # Remove any rows with NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # First 4 columns are features (BMI, AGE, Income, PhysHlth)
    X = df.iloc[:, :4].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    
    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ensure target is binary and convert to float
    y = (y > np.median(y)).astype(np.float32)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=42, stratify=y
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, scaler

def train_model(model, train_loader, test_loader, epochs=50, learning_rate=0.0001):
    """Train the ANFIS model."""
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    best_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0
    min_epochs = 10
    
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
            
            # Check for invalid gradients
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    valid = not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any()
                    if not valid:
                        print(f"Invalid gradients in {name}")
                        valid_gradients = False
                        break
            
            if valid_gradients:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
            else:
                print("Skipping optimization step due to invalid gradients")
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_val, y_val in test_loader:
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item()
                
                predictions = (torch.sigmoid(val_outputs) > 0.5).float()
                correct += (predictions == y_val).sum().item()
                total += y_val.size(0)
                
            val_loss /= len(test_loader)
            accuracy = correct / total
            
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch >= min_epochs and patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

def evaluate_model(model, test_loader):
    """Evaluate the trained model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            outputs = model(X_test)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_test).sum().item()
            total += y_test.size(0)
    
    accuracy = correct / total
    print(f'\nTest Accuracy: {accuracy:.4f}')
    return accuracy

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load and prepare data
    train_loader, test_loader, scaler = prepare_data()
    
    # Create ANFIS model with minimal complexity
    input_dim = 4  # Number of features (BMI, AGE, Income, PhysHlth)
    mf_count = 2  # Number of membership functions per input
    model = ANFISNetwork(input_dim=input_dim, mf_count=mf_count)
    
    # Get a sample batch for initialization
    X_sample = next(iter(train_loader))[0]
    
    # Initialize membership functions based on data
    model.initialize_memberships(X_sample, spread_scale=1.0)
    
    # Train the model
    model = train_model(model, train_loader, test_loader, epochs=50)
    
    # Evaluate the model
    final_accuracy = evaluate_model(model, test_loader)
    
    # Save the model if accuracy is reasonable
    if final_accuracy > 0.6:
        torch.save(model.state_dict(), 'anfis_model.pth')
        print("\nModel saved to 'anfis_model.pth'")

if __name__ == "__main__":
    main()