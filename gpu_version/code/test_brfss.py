"""Test script for BRFSS dataset loading and ANFIS training."""

import os
import torch
from dataset_loader import BRFSSDatasetLoader
from model import ANFISNetwork
from trainer import train_anfis
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare BRFSS dataset
    # Using the provided 300-sample BRFSS CSV (no header)
    dataset_path = "../input/binary_no-head_BMI-AGE-Income-PhysHlth_300x5.csv"
    loader = BRFSSDatasetLoader(
        dataset_path,
        normalization="minmax",  # MinMax scaling works better for bounded features like BMI
        test_size=0.2,
        val_size=0.1,
    )

    # Load and prepare data
    splits = loader.load()
    dataloaders = splits.to_dataloaders(
        batch_size=32,
        shuffle_train=True,
        pin_memory=True
    )

    # Initialize model
    input_size = 4  # BMI, Age, Income, PhysHlth
    num_rules = 4   # Start with 4 rules and adjust based on performance
    model = ANFISNetwork(
        input_dim=input_size,
        mf_count=num_rules,
    ).to(device)

    # Train model
    history = train_anfis(
        model=model,
        dataloaders=dataloaders,
        device=device,
        epochs=50,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        classification=True
    )

    print("\nTraining completed!")
    print(f"Final training loss: {history.train_loss[-1]:.4f}")
    if history.val_loss:
        print(f"Final validation loss: {history.val_loss[-1]:.4f}")
    if history.train_accuracy:
        print(f"Final training accuracy: {history.train_accuracy[-1]:.2%}")
    if history.val_accuracy:
        print(f"Final validation accuracy: {history.val_accuracy[-1]:.2%}")

    # Save trained model
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.abspath(os.path.join(out_dir, "trained_anfis_brfss_300.pth"))
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to: {model_path}")

    # Evaluate on test split if available
    if "test" in dataloaders:
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in dataloaders["test"]:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds = (out.detach().cpu().numpy() > 0.5).astype(int).ravel()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.cpu().numpy().ravel().tolist())

        print("\nEvaluation on test set:")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()