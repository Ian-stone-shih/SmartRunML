import pandas as pd
from src.preprocessing import preprocess

def test_preprocess_output_shape():
    # Sample input data
    data = {
        "Date": ["2023-01-01  11:32:12", "2023-01-02  11:32:12", "2023-01-03  11:32:12"],
        "Title": ["Aachen Activity", "Taichung Activity", "Bristol Activity"],
        "Distance": [1, 2, 3],
        "Body Battery": [60, 70, 80],
        "Sleep": [6, 7, 8],
        "stress": [30, 40, 50],
        "Total Ascent": [100, 200, 300],
        "Total Descent": [90, 180, 270],
        "Avg Pace": ["5:15", "5:34", "5:10"],
        "Calories": [200, 250, 220],
    }
    df = pd.DataFrame(data)

    X_scaled, y_scaled = preprocess(df)

    # Check shapes
    assert X_scaled.shape == (453, 7)
    assert y_scaled.shape == (453, 2)