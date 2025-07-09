# src/smartrunml/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from model import SmartRunNN
import joblib

def train_model(X_train, y_train, X_test, y_test):
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    print(X_train_t.shape, y_train_t.shape, X_test_t.shape, y_test_t.shape)

    model = SmartRunNN(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    losses = []


    for epoch in range(10):
        for i in range(0, len(X_train_t)):

            optimizer.zero_grad()
            predictions = model(X_train_t[i])
            loss = criterion(predictions, y_train_t[i])
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Loss: {loss.item():.4f}')
            losses.append(loss.item())

        with torch.no_grad():
            mse_loss = 0.0
            for i in range(len(X_test_t)):
                model.eval()
                predictions = model(X_test_t[i])
                mse = mean_squared_error(y_test_t[i].numpy(), predictions.numpy())
                mse_loss += mse
            MSE = mse_loss / len(X_test_t)

            
            
        print(f'MSE: {MSE:.4f}')
    

    return model, MSE, losses
