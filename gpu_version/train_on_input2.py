"""Training script for BRFSS diabetes dataset from input2 folder."""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from dataset_loader import BRFSSDatasetLoader
from model import ANFISNetwork
from trainer import train_anfis


def prepare_input2_data(filepath, sample_size=5000, test_size=0.2, val_size=0.1, batch_size=32):
    """Load and prepare the diabetes dataset from input2."""
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Diabetes_binary'].value_counts()}\n")
    
    # Sample for faster training (keep class balance)
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} records to balance classes...")
        # Stratified sample to maintain class balance
        df = df.groupby('Diabetes_binary', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), sample_size // 2), random_state=42)
        )
        print(f"Sampled dataset shape: {df.shape}\n")
    
    # Separate features and target
    X = df.drop('Diabetes_binary', axis=1).values.astype(np.float32)
    y = df['Diabetes_binary'].values.astype(np.float32)
    
    # Scale features to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=42, stratify=y
    )
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train set: {len(X_train)} | Val set: {len(X_val)} | Test set: {len(X_test)}")
    print(f"Number of features: {X_scaled.shape[1]}\n")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, scaler


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Dataset path
    dataset_path = "d:/code_space/Anfis/input2/diabetes_binary_health_indicators_BRFSS2015.csv"
    
    # Load data
    dataloaders, scaler = prepare_input2_data(
        dataset_path,
        sample_size=5000,  # Use 5000 samples for faster training
        batch_size=64
    )
    
    # Model configuration
    input_dim = 21  # 22 columns - 1 target = 21 features
    num_rules = 3   # Start with 3 membership functions per feature
    
    model = ANFISNetwork(
        input_dim=input_dim,
        mf_count=num_rules,
    ).to(device)
    
    print(f"Model created with {input_dim} inputs and {num_rules} membership functions per input")
    print(f"Total rules: {num_rules ** input_dim}\n")
    
    # Train model
    print("Starting training...")
    history = train_anfis(
        model=model,
        dataloaders=dataloaders,
        device=device,
        epochs=30,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        classification=True
    )
    
    # Print training summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Final training loss: {history.train_loss[-1]:.4f}")
    print(f"Final validation loss: {history.val_loss[-1]:.4f}")
    print(f"Final training accuracy: {history.train_accuracy[-1]:.2%}")
    print(f"Final validation accuracy: {history.val_accuracy[-1]:.2%}\n")
    
    # Save model
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.abspath(os.path.join(out_dir, "trained_anfis_input2.pth"))
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}\n")
    
    # Evaluate on test set
    if "test" in dataloaders:
        print("="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        
        model.eval()
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        with torch.no_grad():
            for xb, yb in dataloaders["test"]:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                
                # Get probabilities
                proba = out.detach().cpu().numpy().ravel()
                y_pred_proba.extend(proba.tolist())
                
                # Get binary predictions (threshold 0.5)
                preds = (proba > 0.5).astype(int)
                y_pred.extend(preds.tolist())
                y_true.extend(yb.cpu().numpy().ravel().astype(int).tolist())
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test F1-Score: {f1:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print(f"\nTrue Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        # Save evaluation results
        results_path = os.path.abspath(os.path.join(out_dir, "eval_results_input2.txt"))
        with open(results_path, 'w') as f:
            f.write("ANFIS Training on BRFSS Diabetes Dataset (input2)\n")
            f.write("="*60 + "\n\n")
            f.write("TRAINING SUMMARY\n")
            f.write("-"*60 + "\n")
            f.write(f"Final training loss: {history.train_loss[-1]:.4f}\n")
            f.write(f"Final validation loss: {history.val_loss[-1]:.4f}\n")
            f.write(f"Final training accuracy: {history.train_accuracy[-1]:.2%}\n")
            f.write(f"Final validation accuracy: {history.val_accuracy[-1]:.2%}\n\n")
            
            f.write("TEST SET EVALUATION\n")
            f.write("-"*60 + "\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test F1-Score: {f1:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            f.write(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}\n")
            f.write(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}\n\n")
            
            f.write(classification_report(y_true, y_pred, digits=4))
        
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
