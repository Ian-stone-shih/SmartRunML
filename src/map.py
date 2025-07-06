import requests
import openrouteservice
from openrouteservice import convert
import folium
import matplotlib.pyplot as plt
from geopy.distance import geodesic

def geocode_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "SmartRunML/1.0 (your_email@example.com)"}
    response = requests.get(url, params=params, headers=headers)

    # Check status
    response.raise_for_status()  # Will raise HTTPError if 4xx/5xx

    results = response.json()
    if results:
        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])
        return lon, lat
    else:
        raise ValueError("Address not found")


def route_plan(API_key, lon, lat):
    # Initialize client
    client = openrouteservice.Client(
        key=API_key,
    )

    # Set starting location: [longitude, latitude]
    start_coords = [lon, lat]  # Example: Aachen, Germany

    # Parameters for a 5 km circular running route
    params = {
        "coordinates": [start_coords],
        "profile": "foot-walking",
        "format_out": "geojson",
        "format": "geojson",
        "options": {
            "round_trip": {
                "length": 5000,  # meters
                "seed": 2,  # random seed to get different variations
            }
        },
    }

    # Request route
    route_geojson = client.directions(**params)

    # Extract coordinates
    coordinates = route_geojson["features"][0]["geometry"]["coordinates"]

    # Folium expects (lat, lon), so we reverse each pair
    latlon_coords = [[lat, lon] for lon, lat in coordinates]

    # Center of the map
    center = latlon_coords[0]

    # Create a map centered on your starting point
    m = folium.Map(location=center, zoom_start=15)

    # Add the route as a polyline
    folium.PolyLine(latlon_coords, color="blue", weight=4, opacity=0.8).add_to(m)

    # Add a marker at the start
    folium.Marker(center, tooltip="Start").add_to(m)

    m

    return coordinates, m

def elevation(coordinates):
    # Open-Elevation API expects lat/lon
    locations = [{"latitude": lat, "longitude": lon} for lon, lat in coordinates]

    # Prepare request body
    request_json = {"locations": locations}

    # Make request
    response = requests.post(
        "https://api.open-elevation.com/api/v1/lookup", json=request_json
    )

    # Parse response
    elevations = [result["elevation"] for result in response.json()["results"]]

    # Compute distances between points
    distances = [0.0]
    for i in range(1, len(coordinates)):
        p1 = (coordinates[i - 1][1], coordinates[i - 1][0])  # (lat, lon)
        p2 = (coordinates[i][1], coordinates[i][0])
        d = geodesic(p1, p2).meters
        distances.append(distances[-1] + d)


    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(distances, elevations, marker="o")
    plt.xlabel("Distance along route (meters)")
    plt.ylabel("Elevation (meters)")
    plt.title("Elevation Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return elevations

def des_asc(elevations):
    # Calculate total ascent and descent
    total_ascent = 0
    total_descent = 0

    for i in range(1, len(elevations)):
        if elevations[i] > elevations[i - 1]:
            total_ascent += elevations[i] - elevations[i - 1]
        elif elevations[i] < elevations[i - 1]:
            total_descent += elevations[i - 1] - elevations[i]

    return total_ascent, total_descent
    