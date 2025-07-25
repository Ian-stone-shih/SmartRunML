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
from src.preprocessing import format_pace

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .stSlider > div {
        padding-top: 0.1rem;
        padding-bottom: 0.1rem;
    }
    section[data-testid="stSidebar"] .stButton button {
        padding: 0.1rem 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Route Planner")
    # Address input
    address = st.text_input("Start Address", "Aachen, Germany")

    col_distance, col_button = st.columns([2, 1])
    with col_distance:
        # Distance input
        distance_km = st.slider("Distance (km)", 1, 21, 5)
    with col_button:
        if "route_seed" not in st.session_state:
            st.session_state.route_seed = 1

        if st.button("New Route"):
            st.session_state.route_seed += 1


# --- Geocode ---
geolocator = Nominatim(user_agent="smart_run_app")
location = geolocator.geocode(address)
start_coords = [location.longitude, location.latitude]

# --- route_seed---
seed = st.session_state.route_seed

# --- OpenRouteService API Key ---
load_dotenv()  # load .env file
ORS_API_KEY = os.environ["ORS_API_KEY"]

# --- OpenRouteService ---
coords, m = route_plan(ORS_API_KEY, location.longitude, location.latitude, distance_km, seed)

# --- Elevation Profile ---
from src.map import elevation
elevations, distances = elevation(coords)
total_ascent, total_descent = des_asc(elevations)
# --- Current Temperature ---
Current_t = get_temp_meteostat(location.latitude, location.longitude, datetime.now())

st.markdown(
    """
    <style>
    /* Reduce padding/margin between elements */
    section[data-testid="stSidebar"] > div > div {
        gap: 0rem;
    }
    /* Reduce padding above and below each widget */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    /* Reduce vertical spacing inside sliders */
    section[data-testid="stSidebar"] .stSlider > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Sidebar Inputs ---
with st.sidebar:
    # Ascent and descent side by side
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ascent (m)", f"{total_ascent:.1f}")
    with col2:
        st.metric("Descent (m)", f"{total_descent:.1f}")      
    st.metric("Current Temperature (°C)", f"{Current_t:.1f}")

    # Body Battery, Sleep Hours, Stress Level vertically
    st.subheader("Your Condition")
    body_battery = st.slider("Body Battery (%)", 0, 100, 60)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
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

pace = format_pace(predictions[0][0])
calories = predictions[0][1]

# Plot

col_map, col_elev = st.columns([2, 1])  # Adjust ratio as needed

with col_map:
    st.subheader("Route Map")
    st_folium(m, width=700, height=500)

with col_elev:
    st.subheader("Elevation Profile")
    # Plot elevation
    fig, ax = plt.subplots()
    ax.plot(distances, elevations)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Elevation Profile")
    st.pyplot(fig)

    st.subheader("Predicted Performance")
    st.metric("Predicted Pace (min/km)", pace)
    st.metric("Estimated Calories", f"{calories:.0f} kcal")

