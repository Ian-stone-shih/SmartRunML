import streamlit as st
import folium
from folium import Map, PolyLine
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import openrouteservice
import os
from dotenv import load_dotenv

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
client = openrouteservice.Client(key=ORS_API_KEY)
params = {
    "coordinates": [start_coords],
    "profile": "foot-walking",
    "format_out": "geojson",
    "options": {
        "round_trip": {"length": distance_km * 1000, "seed": 1}
    }
}
route = client.directions(**params)
coords = route["features"][0]["geometry"]["coordinates"]

# --- Folium Map ---
# Folium expects (lat, lon), so we reverse each pair
latlon_coords = [[lat, lon] for lon, lat in coords]
# Center of the map
center = latlon_coords[0]

# Create a map centered on your starting point
m = folium.Map(location=center, zoom_start=15)

# Add the route as a polyline
folium.PolyLine(latlon_coords, color="blue", weight=4, opacity=0.8).add_to(m)

# Add a marker at the start
folium.Marker(center, tooltip="Start").add_to(m)


# Show map
st_folium(m, width=700)

# --- Elevation Profile ---
# Fetch elevation (using Open Elevation API)
locations = [{"latitude": lat, "longitude": lon} for lon, lat in coords]
elev_req = requests.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": locations})
elevations = [x["elevation"] for x in elev_req.json()["results"]]

# Compute cumulative distance
dists = [0.0]
for i in range(1, len(coords)):
    p1 = (coords[i-1][1], coords[i-1][0])
    p2 = (coords[i][1], coords[i][0])
    d = geodesic(p1, p2).meters
    dists.append(dists[-1] + d)

# Plot elevation
fig, ax = plt.subplots()
ax.plot(dists, elevations)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Elevation Profile")
st.pyplot(fig)
