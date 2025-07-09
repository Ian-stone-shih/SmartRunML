# src/smartrunml/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
import datetime
from meteostat import Hourly, Point
from geopy.geocoders import Nominatim
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

geolocator = Nominatim(user_agent="smart_runml")


# Function to get coordinates from city name
def geocode_city(city_name):
    try:
        location = geolocator.geocode(city_name)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print("Geocoding error:", e)
    return (None, None)


# Function to get temperature from Meteostat
def get_temp_meteostat(lat, lon, dt):
    location = Point(lat, lon)
    start = dt.replace(minute=0, second=0, microsecond=0)
    end = dt.replace(minute=0, second=0, microsecond=0)
    data = Hourly(location, start, end).fetch()
    if data.empty:
        return None
    return data["temp"].iloc[0]


def convert_pace_to_seconds(pace_str):
    """Convert 'MM:SS' string to total seconds."""
    try:
        minutes, seconds = map(int, pace_str.split(":"))
        return minutes * 60 + seconds
    except:
        return None  # if invalid format

def analysis_data(df):
    """Visualize the data distribution and correlations."""
    features = ["Distance", "Body Battery", "Sleep", "stress", "Total Ascent", "Total Descent", "Temperature"]
    target = "Avg Pace"
    # Keep only the columns we need
    df = df[features + [target]]
    # 1. Summary statistics
    #print(df.describe())

    # 2. Correlation matrix
    corr = df.corr()
    #print("\nCorrelation matrix:\n", corr)
    print(corr.loc["Avg Pace"])

    # 3. Heatmap of correlations
    #plt.figure(figsize=(10,8))
    #sns.heatmap(corr, annot=True, cmap="coolwarm")
    #plt.title("Feature Correlations")
    #plt.show()

    # 4. Histograms
    #df.hist(bins=15, figsize=(15,10))
    #plt.tight_layout()
    #plt.show()
    #print(df.head())

def synthetic_data(df_clean, features, num_samples):


    synthetic_rows = []

    for _ in range(num_samples):
        # Randomly pick a row
        base = df_clean.sample(1).iloc[0]

        # Add small noise
        row = base.copy()
        for col in features[:-1]:
            noise = np.random.normal(-0.03, 0.03)  # ~5% noise
            row[col] = row[col] * (1 + noise)

        # Optionally perturb Avg Pace proportionally to Sleep and Stress
        pace_adjust = 0.5*(row["Sleep"] - base["Sleep"]) * (-0.55) + (row["Temperature"] - base["Temperature"]) * (-0.43)
        row["Avg Pace"] = base["Avg Pace"] + pace_adjust 

        synthetic_rows.append(row)

    # Combine into DataFrame
    df_synth = pd.DataFrame(synthetic_rows)
    df_combined = pd.concat([df_clean, df_synth], ignore_index=True)
    
    return df_combined


def preprocess(df):
    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    # Fetch temperatures
    temps = []
    for _, row in df.iterrows():
        city = row["Title"].split()[0]
        lat, lon = geocode_city(city)
        dt = row["Date"]
        if lat is None:
            temps.append(None)
            continue
        temp = get_temp_meteostat(lat, lon, dt)
        temps.append(temp)
        time.sleep(1)  # Avoid Meteostat rate limits

    df["Temperature"] = temps

    features = ["Distance", "Body Battery", "Sleep", "stress", "Total Ascent", "Total Descent", "Temperature"]
    target = "Avg Pace"
    # Keep only the columns we need
    df = df[features + [target]]

    #df["Steps"] = df["Steps"].astype(str).str.replace(",", "")
    df["Avg Pace"] = df["Avg Pace"].apply(convert_pace_to_seconds)
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()  # Drop rows with NaN values
    analysis_data(df)  # Perform analysis on the data
    df_sys = synthetic_data(df, features, 300)  # Add synthetic data
    analysis_data(df_sys)  # Perform analysis again after adding synthetic data
    # Split features and target
    X = df_sys[features].values
    y = df_sys[target].values

    # Scale X
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale y
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    joblib.dump(scaler_X, "scaler_X.save")
    joblib.dump(scaler_y, "scaler_y.save")

    return train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)

