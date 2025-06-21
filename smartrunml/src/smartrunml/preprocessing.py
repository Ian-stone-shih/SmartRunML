# src/smartrunml/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):
    df = pd.read_csv(path)

    features = ['Distance', 'Steps', 'Calories', 'Total Ascent', 'Total Descent']
    target = 'Avg Pace'
        # Keep only the columns we need
    df = df[features + [target]]

    # Convert all selected columns to numeric (invalid values become NaN)
    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    

    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
