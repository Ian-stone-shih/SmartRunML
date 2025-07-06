import requests

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