from src.preprocessing import preprocess
from src.train import train_model
from src.train import train_model_kfold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("data/Activities-6-2-2.csv")

# 2. Preprocess
X_scaled, y_scaled = preprocess(df)

# 3. Train model
model, train_losses, val_losses, test_mse = train_model(X_scaled, y_scaled)
# print("MSE:", test_mse)

# K-fold training
#train_losses, fold_losses = train_model_kfold(X_scaled, y_scaled, 5)

# 4. Plot training and validation loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss over Epochs")
# Add annotation for test loss
plt.text(
    0.7,
    0.5,
    "Test MSE Loss: {:.4f}".format(test_mse),
    fontsize=12,
    ha="center",
    va="center",
    transform=plt.gca().transAxes,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)
plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Save the model
torch.save(model.state_dict(), "model/final_model.pt")

# 6. Prepare new input
X_new = np.array([[5, 30, 470, 30, 100, 100, 30]])
scaler_X = joblib.load("scaler_X.save")
scaler_y = joblib.load("scaler_y.save")

X_new_scaled = scaler_X.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# 7. Predict
model.eval()  # Important: set to eval mode
with torch.no_grad():
    y_pred_scaled = model(X_new_tensor).numpy()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

print("Predicted:", y_pred_original)
