# src/smartrunml/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from model import SmartRunNN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import joblib

def train_model(X_scaled, y_scaled):
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)
    # Split train and validation (e.g., 15% of remaining as validation)
    print(X_temp.shape, X_test.shape, y_temp.shape, y_test.shape)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42
    )
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = SmartRunNN(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t)
            val_losses.append(val_loss.item())

        print(
            f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}"
        )
    # Evaluate on test set
    model.eval()

    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t)
    print(f"Test Loss = {test_loss.item():.4f}")
                
                
    

    return model, train_losses, val_losses, test_loss.item()

def train_model_kfold(X, y, k_folds):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # To track results
    fold_losses = []
    train_losses = []

    # For demonstration, save models
    saved_model_paths = []

    # Fold counter
    fold = 1

    for train_idx, val_idx in kf.split(X):
        print(f"\n========== Fold {fold}/{k_folds} ==========")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        # Initialize model
        model = SmartRunNN(input_size=X.shape[1])

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}")
                train_losses.append(loss.item())

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)

        print(f"Fold {fold} Validation Loss: {val_loss.item():.4f}")
        fold_losses.append(val_loss.item())

        # Save model
        model_path = f"smart_run_fold_{fold}.pt"
        torch.save(model.state_dict(), model_path)
        saved_model_paths.append(model_path)
        print(f"Saved model to {model_path}")

        fold += 1

    return fold_losses, train_losses