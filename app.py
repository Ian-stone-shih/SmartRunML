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

# --- Sidebar Inputs ---
st.sidebar.title("Route Planner")
address = st.sidebar.text_input("Start Address", "Aachen, Germany")
distance_km = st.sidebar.slider("Distance (km)", 1, 20, 5)


# --- Geocode ---
geolocator = Nominatim(user_agent="smart_run_app")
location = geolocator.geocode(address)
start_coords = [location.longitude, location.latitude]

# --- OpenRouteService API Key ---
load_dotenv()  # load .env file
ORS_API_KEY = os.environ["ORS_API_KEY"]

# --- OpenRouteService ---
coords, m = route_plan(ORS_API_KEY, location.longitude, location.latitude, distance_km)

# Show map
st_folium(m, width=700)

# --- Elevation Profile ---
from src.map import elevation
elevations, distances = elevation(coords)
total_ascent, total_descent = des_asc(elevations)
# --- Current Temperature ---
Current_t = get_temp_meteostat(location.latitude, location.longitude, datetime.now())

# --- Calculate Ascent and Descent ---
with st.sidebar:
    st.metric("Total Ascent (m)", f"{total_ascent:.1f}")
    st.metric("Total Descent (m)", f"{total_descent:.1f}")
    st.metric("Current Temperature (°C)", f"{Current_t:.1f}")


# Plot
# Plot elevation
fig, ax = plt.subplots()
ax.plot(distances, elevations)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Elevation Profile")
st.pyplot(fig)
