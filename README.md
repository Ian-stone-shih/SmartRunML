# SmartRunML

**SmartRunML** is a Python package that predicts personalized running plans using **Garmin data** and **weather conditions**.  
It uses a neural network model to output **tailored running pace** and **estimated calories burned** based on user input.

This project also includes a **Streamlit-based web app** for:
- Route map generation  
- Elevation profile visualization  
- Predictions based on your own physiological and route data

---

## How to Use SmartRunML

SmartRunML helps you build personalized running plans using your own Garmin data and launch a web-based app to visualize predictions.

---

### 1. Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/Ian-stone-shih/smartrunml.git
cd smartrunml
pip install -r requirements.txt
```

---

### 2. Configure Your Dataset

Create or modify the provided `config.yaml` file in the root directory:

```yaml
data_path: "src/data/Activities-6.csv"   # path to your CSV file
synthetic_data_size: 450                 # number of synthetic samples to generate
```

**Note**: In Garmin, features like `Body Battery`, `Sleep`, and `Stress` are stored separately from running activity data.  
> Your CSV file must contain columns like: Distance, Sleep, Body Battery, Stress, Ascent, Descent, Temperature, Avg Pace, Calories. You are responsible for **manually merging these values** with your running dataset.  

---

### 3. Train Your Model

Run the following script to preprocess the data, generate synthetic samples, and train the neural network:

```bash
python main.py
```

This will:
- Load your data
- Add synthetic points
- Normalize features
- Train using different optimizers
- Save scalers and the trained model

**Note**: Generate synthetic samples if you don't have enough data.

---

### 4. Launch the Web App

After training, launch the Streamlit app with:

```bash
streamlit run app.py
```

You can now:
- Input values like Distance, Sleep, Stress, etc.
- Visualize the predicted pace and calories
- See a map and elevation profile for route planning

---

