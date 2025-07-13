import streamlit as st
from datetime import datetime, timedelta
from meteostat import Point, Hourly
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv
from src.map import route_plan
from src.map import des_asc
from src.preprocessing import get_temp_meteostat
import joblib
import torch
import torch.nn as nn

# --- Sidebar Inputs ---
st.sidebar.title("Route Planner")
address = st.sidebar.text_input("Start Address", "Aachen, Germany")
distance_km = st.sidebar.slider("Distance (km)", 1, 20, 5)


# --- Geocode ---
geolocator = Nominatim(user_agent="smart_run_app")
location = geolocator.geocode(address)
start_coords = [location.longitude, location.latitude]

# --- route_seed---
# Initialize seed counter in session_state
if "route_seed" not in st.session_state:
    st.session_state.route_seed = 1

if st.sidebar.button("Generate New Route"):
    st.session_state.route_seed += 1

seed = st.session_state.route_seed

# --- OpenRouteService API Key ---
load_dotenv()  # load .env file
ORS_API_KEY = os.environ["ORS_API_KEY"]

# --- OpenRouteService ---
coords, m = route_plan(ORS_API_KEY, location.longitude, location.latitude, distance_km, seed)

# Show map
st_folium(m, width=700)

# --- Elevation Profile ---
from src.map import elevation
elevations, distances = elevation(coords)
total_ascent, total_descent = des_asc(elevations)
# --- Current Temperature ---
Current_t = get_temp_meteostat(location.latitude, location.longitude, datetime.now())

# --- Sidebar Inputs ---
with st.sidebar:
    st.metric("Total Ascent (m)", f"{total_ascent:.1f}")
    st.metric("Total Descent (m)", f"{total_descent:.1f}")
    st.metric("Current Temperature (°C)", f"{Current_t:.1f}")

    # Group sliders in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        body_battery = st.slider("Body Battery (%)", 0, 100, 75)
    with col2:
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    with col3:
        stress_level = st.slider("Stress Level (1-100)", 1, 100, 40)


# --- Load Model ---
class MySmartRunNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)


input_features = [
    distance_km,
    body_battery,
    sleep_hours * 60,  # Convert sleep hours to minutes
    stress_level,
    total_ascent,
    total_descent,
    Current_t
]
X_input = [input_features]
scaler_X = joblib.load("src/scaler_X.save")
scaler_y = joblib.load("src/scaler_y.save")

X_new_scaled = scaler_X.transform(X_input)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

model = MySmartRunNN(input_size=7)
model.load_state_dict(torch.load("model/final_model.pt"))
model.eval()
# Predict
with torch.no_grad():
    y_pred = model(X_new_tensor).numpy()
    predictions = scaler_y.inverse_transform(y_pred)

pace = predictions[0][0]
calories = predictions[0][1]

# Plot
# Plot elevation
fig, ax = plt.subplots()
ax.plot(distances, elevations)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Elevation Profile")
st.pyplot(fig)

st.subheader("Predicted Performance")
st.write(f"Predicted Pace: **{pace:.2f} seconds/km**")
st.write(f"Estimated Calories Burned: **{calories:.0f} kcal**")