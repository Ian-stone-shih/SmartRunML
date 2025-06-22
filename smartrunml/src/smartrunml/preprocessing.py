# src/smartrunml/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def convert_pace_to_seconds(pace_str):
    """Convert 'MM:SS' string to total seconds."""
    try:
        minutes, seconds = map(int, pace_str.split(":"))
        return minutes * 60 + seconds
    except:
        return None  # if invalid format


def load_and_preprocess(path):
    df = pd.read_csv(path)

    features = ["Distance", "Steps", "Calories", "Total Ascent", "Total Descent"]
    target = "Avg Pace"
    # Keep only the columns we need
    df = df[features + [target]]

    df["Steps"] = df["Steps"].astype(str).str.replace(",", "")
    df["Avg Pace"] = df["Avg Pace"].apply(convert_pace_to_seconds)
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    print(df.head())
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

