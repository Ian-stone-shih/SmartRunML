import streamlit as st

from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv
from src.map import route_plan

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

# Plot
# Plot elevation
fig, ax = plt.subplots()
ax.plot(distances, elevations)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Elevation Profile")
st.pyplot(fig)
