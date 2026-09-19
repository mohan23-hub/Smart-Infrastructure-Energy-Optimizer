import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# SECTION 1: DATA SYNTHESIS & MACHINE LEARNING PIPELINE
# ==============================================================================


def generate_building_data(days=90):
    """Generates synthetic time-series data for smart building energy usage."""
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=days * 24, freq="h")
    df = pd.DataFrame({"timestamp": dates})

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Outdoor Temperature (°C)
    df["outdoor_temp"] = 22 + 8 * np.sin(
        (df["hour"] - 9) * np.pi / 12
    ) + np.random.normal(0, 1.5, len(df))

    # Building Occupancy (0 to 100 people)
    occupancy_base = np.where(
        df["is_weekend"] == 1,
        5,
        np.where((df["hour"] >= 8) & (df["hour"] <= 18), 75, 10),
    )
    df["occupancy"] = np.clip(
        occupancy_base + np.random.normal(0, 10, len(df)), 0, 100
    ).astype(int)

    # Dynamic Pricing in INR (₹/kWh) -> ₹12 Peak vs ₹6 Off-Peak
    df["electricity_rate"] = np.where(
        (df["hour"] >= 14) & (df["hour"] <= 19), 12.0, 6.0
    )

    # True Energy Load Calculation (kWh)
    base_load = 20.0
    hvac_load = np.maximum(0, df["outdoor_temp"] - 20) * 3.5
    occupancy_load = df["occupancy"] * 0.4
    noise = np.random.normal(0, 2.0, len(df))

    df["energy_kwh"] = base_load + hvac_load + occupancy_load + noise
    return df


@st.cache_resource
def train_and_get_model():
    """Trains a Random Forest regressor natively."""
    df = generate_building_data(days=90)
    df["energy_lag_1h"] = df["energy_kwh"].shift(1)
    df["temp_rolling_3h"] = df["outdoor_temp"].rolling(3).mean()
    df = df.dropna()

    features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "outdoor_temp",
        "occupancy",
        "energy_lag_1h",
        "temp_rolling_3h",
    ]
    X = df[features]
    y = df["energy_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train Native Scikit-Learn Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2


# ==============================================================================
# SECTION 2: STREAMLIT DASHBOARD INTERFACE
# ==============================================================================

st.set_page_config(
    page_title="Smart Infrastructure ML Optimizer", layout="wide"
)

st.title("🌱 Sustainable Smart Infrastructure Energy Optimizer")
st.caption(
    "Predictive Machine Learning Framework for Dynamic Building Energy"
    " Management"
)

# Load Model
model, mae, r2 = train_and_get_model()

st.success(
    f"✅ Native ML Model Trained Successfully! Evaluation -> MAE: {mae:.2f} kWh"
    f" | R² Score: {r2:.2f}"
)

# Sidebar Controls (Mocking Real-time IoT Sensor Feed)
st.sidebar.header("🕹️ Simulated Sensor Inputs")
sim_hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
sim_temp = st.sidebar.slider("Outdoor Temperature (°C)", 15.0, 42.0, 32.0)
sim_occ = st.sidebar.slider("Building Occupancy Count", 0, 100, 65)

# Indian Rupee Utility Tariff Options
grid_rate_str = st.sidebar.radio(
    "Grid Utility Tariff",
    ["Off-Peak (₹6.00/kWh)", "Peak Demand (₹12.00/kWh)"],
)

rate_value = 12.0 if "Peak" in grid_rate_str else 6.0

# Construct Input Data Frame for Inference
input_data = pd.DataFrame(
    [
        {
            "hour": sim_hour,
            "day_of_week": 2,  # Wednesday
            "is_weekend": 0,
            "outdoor_temp": sim_temp,
            "occupancy": sim_occ,
            "energy_lag_1h": 45.0,
            "temp_rolling_3h": sim_temp - 0.5,
        }
    ]
)

# Predict Baseline Consumption
predicted_baseline_kwh = float(model.predict(input_data)[0])

# Optimization Logic Engine
optimized_kwh = predicted_baseline_kwh
savings_percent = 0.0

if sim_occ == 0:
    optimized_kwh *= 0.40  # Eco mode for unoccupied space
    savings_percent = 60.0
elif rate_value > 8.0 and sim_occ < 30:
    optimized_kwh *= 0.75  # Throttling during expensive pricing
    savings_percent = 25.0
elif sim_occ < 50:
    optimized_kwh *= 0.85
    savings_percent = 15.0

# Render Metrics Grid with Rupee Symbol
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Baseline Load", f"{predicted_baseline_kwh:.2f} kWh")
col2.metric(
    "Optimized Smart Load",
    f"{optimized_kwh:.2f} kWh",
    delta=f"-{savings_percent:.1f}% Reduction",
)
col3.metric(
    "Unoptimized Hourly Cost",
    f"₹{(predicted_baseline_kwh * rate_value):.2f}",
)
col4.metric(
    "ML Optimized Hourly Cost", f"₹{(optimized_kwh * rate_value):.2f}"
)

st.divider()

# Interactive Profile Plotting
st.write("### 📊 24-Hour Predictive Demand Profile")
hours = np.arange(24)
baseline_profile = [
    predicted_baseline_kwh * (0.6 + 0.4 * np.sin((h - 8) * np.pi / 12))
    for h in hours
]
optimized_profile = [
    val * (0.70 if 14 <= h <= 19 else 0.90)
    for h, val in zip(hours, baseline_profile)
]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hours,
        y=baseline_profile,
        name="Traditional HVAC Schedule",
        line=dict(color="red", dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=hours,
        y=optimized_profile,
        name="ML Smart Optimized Schedule",
        line=dict(color="green", width=3),
    )
)
fig.update_layout(
    title="Daily Energy Consumption Curve (kWh)",
    xaxis_title="Hour of Day",
    yaxis_title="Energy Usage (kWh)",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)