import pytest
from unittest.mock import patch
from src.map import geocode_address, des_asc, elevation

# ✅ Test geocode_address (success case)
@patch("src.map.requests.get")
def test_geocode_address_success(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [{"lat": "50.77", "lon": "6.08"}]

    lon, lat = geocode_address("Aachen")
    assert abs(lat - 50.77) < 0.01
    assert abs(lon - 6.08) < 0.01

# ✅ Test geocode_address (fail case)
@patch("src.map.requests.get")
def test_geocode_address_failure(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = []

    with pytest.raises(ValueError, match="Address not found"):
        geocode_address("Fake Place")

# ✅ Test des_asc (pure function, no mock)
def test_des_asc_logic():
    elevations = [10, 20, 15, 25, 22]
    ascent, descent = des_asc(elevations)
    assert ascent == 20  # 10→20 and 15→25
    assert descent == 8  # 20→15 and 25→22

# ✅ Test elevation with mock
@patch("src.map.requests.post")
def test_elevation_success(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "results": [{"elevation": 10}, {"elevation": 12}, {"elevation": 8}]
    }

    coords = [[6.08, 50.77], [6.09, 50.78], [6.10, 50.79]]
    elevations, distances = elevation(coords)
    assert elevations == [10, 12, 8]
    assert len(distances) == 3