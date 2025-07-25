from src.preprocessing import preprocess
from src.train import train_model
from src.train import train_model_kfold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import time

# 1. Load Data
df = pd.read_csv("src/data/Activities-6.csv")

# 2. Preprocess
X_scaled, y_scaled = preprocess(df)

# 3. Train model
# SGD
#lr = 0.0025  # Learning rate for SGD
start_1 = time.time()
model_1, train_losses_1, val_losses_1, test_mse_1 = train_model(X_scaled, y_scaled, "ADAM", 0.1)
time_1 = time.time() - start_1

# Adam
start_2 = time.time()
model_2, train_losses_2, val_losses_2, test_mse_2 = train_model(X_scaled, y_scaled, "ADAM", 0.001)
time_2 = time.time() - start_2

# RMSprop
start_3 = time.time()
model_3, train_losses_3, val_losses_3, test_mse_3 = train_model(X_scaled, y_scaled, "ADAM", 0.00001)
time_3 = time.time() - start_3
# print("MSE:", test_mse)

# K-fold training
#train_losses_1, fold_losses = train_model_kfold(X_scaled, y_scaled, 5)

# 1. Create figure
plt.figure(figsize=(10,6))

# 2. Plot training losses
plt.plot(train_losses_1, label="learning rate 0.1 (ADAM)")
#plt.plot(val_losses_1, label="Validation Loss (SGD)", linestyle='--')
plt.plot(train_losses_2, label="learning rate 0.001 (ADAM)")
plt.plot(train_losses_3, label="learning rate 0.00001 (ADAM)")

# 3. Add test loss points
plt.scatter(
    len(train_losses_1)-1,
    test_mse_1,
    color="red",
    marker="o",
    s=60,
    label="Test Loss (0.1)"
)
plt.scatter(
    len(train_losses_2)-1,
    test_mse_2,
    color="green",
    marker="o",
    s=60,
    label="Test Loss (0.001)"
)
plt.scatter(
    len(train_losses_3)-1,
    test_mse_3,
    color="blue",
    marker="o",
    s=60,
    label="Test Loss (0.00001)"
)
'''
# Create the text with times
time_text = (
    f"SGD time: {time_1:.1f}s\n"
    f"Adam time: {time_2:.1f}s\n"
    f"RMSprop time: {time_3:.1f}s"
)

# Add text in figure space (x=0.75 means 75% from the left, y=0.85 means near the top)
plt.gcf().text(
    0.75,     # X position in figure coords
    0.62,     # Y position in figure coords
    time_text,
    fontsize=10,
    ha="left",
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)

time_text = (
    f"SGD: lr={lr} (M), time={time_1:.1f}s\n"
    f"Adam: lr={lr}, time={time_2:.1f}s\n"
    f"RMSprop: lr={lr}, time={time_3:.1f}s"
)

plt.gcf().text(
    0.68, 0.5,
    time_text,
    fontsize=10,
    ha="left",
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)'''
# 5. Final touches
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss and Test MSE for Different learning Rates (RMSprop)")
plt.legend()
plt.grid(True)

# 6. Save and show
plt.savefig("analysis/learning_rate_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Save the model
torch.save(model_2.state_dict(), "model/final_model.pt")

# 6. Prepare new input
X_new = np.array([[5, 30, 470, 30, 100, 100, 30]])
scaler_X = joblib.load("src/scaler_X.save")
scaler_y = joblib.load("src/scaler_y.save")

X_new_scaled = scaler_X.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# 7. Predict
model_2.eval()  # Important: set to eval mode
with torch.no_grad():
    y_pred_scaled = model_2(X_new_tensor).numpy()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

print("Predicted:", y_pred_original)
