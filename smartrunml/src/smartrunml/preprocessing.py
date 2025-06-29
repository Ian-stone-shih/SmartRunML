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


def load_and_preprocess(path):
    df = pd.read_csv(path)
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

    features = ["Distance", "Body Battery", "Sleep", "stress","Temperature", "Total Ascent", "Total Descent"]
    target = "Avg Pace"
    # Keep only the columns we need
    df = df[features + [target]]

    #df["Steps"] = df["Steps"].astype(str).str.replace(",", "")
    df["Avg Pace"] = df["Avg Pace"].apply(convert_pace_to_seconds)
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    # 1. Summary statistics
    print(df.describe())

    # 2. Correlation matrix
    corr = df.corr()
    print("\nCorrelation matrix:\n", corr)

    # 3. Heatmap of correlations
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlations")
    plt.show()

    # 4. Histograms
    df.hist(bins=15, figsize=(15,10))
    plt.tight_layout()
    plt.show()
    print(df.head())

    # Split features and target
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.save")

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

