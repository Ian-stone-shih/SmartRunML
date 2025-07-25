import numpy as np
from src.train import train_model
import torch

def generate_dummy_data(n_samples=100, n_features=7, n_outputs=2):
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.rand(n_samples, n_outputs).astype(np.float32)
    return X, y

def test_train_model_with_all_optimizers():
    X, y = generate_dummy_data()

    for opt in ["SGD", "ADAM", "RMSprop"]:
        model, train_losses, val_losses, test_loss = train_model(X, y, optimizer_type=opt, learning_rate=0.001)

        # Basic structure checks
        assert isinstance(model, torch.nn.Module)
        assert isinstance(train_losses, list) and len(train_losses) == 500
        assert isinstance(val_losses, list) and len(val_losses) == 500
        assert isinstance(test_loss, float)
        assert test_loss >= 0.0