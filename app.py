import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚲",
    layout="centered"
)

# Load model
model = joblib.load("models/final_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Title
st.title("🚲 Bike Rental Demand Predictor")
st.write(
    "Predict the expected number of bike rentals "
    "based on weather, time, and calendar conditions."
)

# User inputs
st.subheader("Enter Conditions")

season = st.selectbox(
    "Season",
    ["Spring", "Summer", "Fall", "Winter"]
)

weather = st.selectbox(
    "Weather",
    [
        "Clear / Partly Cloudy",
        "Mist / Cloudy",
        "Light Rain / Snow",
        "Heavy Rain / Snow"
    ]
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=20.0
)

atemp = st.number_input(
    "Feeling Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=20.0
)

humidity = st.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=60
)

windspeed = st.number_input(
    "Windspeed",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

hour = st.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=18
)

day_of_week = st.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

working_day = st.selectbox(
    "Working Day",
    ["Yes", "No"]
)

# Convert inputs
season_map = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4
}

weather_map = {
    "Clear / Partly Cloudy": 1,
    "Mist / Cloudy": 2,
    "Light Rain / Snow": 3,
    "Heavy Rain / Snow": 4
}

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# Prediction
if st.button("Predict Bike Demand"):

    season_value = season_map[season]
    weather_value = weather_map[weather]
    day_value = day_map[day_of_week]

    is_weekend = int(day_value >= 5)
    peak_hour = int(
        (
            day_value < 5
            and hour in [7, 8, 9, 17, 18, 19]
        )
        or
        (
            day_value >= 5
            and 10 <= hour <= 18
        )
    )

    input_data = pd.DataFrame({
        "season": [season_value],
        "holiday": [0],
        "workingday": [1 if working_day == "Yes" else 0],
        "weather": [weather_value],
        "temp": [temperature],
        "atemp": [atemp],
        "humidity": [humidity],
        "windspeed": [windspeed],
        "year": [2012],
        "month": [6],
        "day": [15],
        "hour": [hour],
        "day_of_week": [day_value],
        "is_weekend": [is_weekend],
        "peak_hour": [peak_hour]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["season", "weather"],
        drop_first=False
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]
    
    prediction = max(0, prediction)
    
    st.success(
        f"Predicted Bike Rentals: {prediction:.0f}"
    )