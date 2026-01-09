import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Daily Temperature Prediction", layout="centered")

st.title("🌤️ Daily Temperature Prediction")
st.write("Linear Regression using Sunlight & Humidity")

st.divider()

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("weather_data.csv")

df = load_data()

# ------------------ FEATURES & TARGET ------------------
X = df[['hours_sunlight', 'humidity_level']]
y = df['daily_temperature']

# ------------------ TRAIN TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ MODEL TRAINING ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ PREDICTION ------------------
y_pred = model.predict(X_test)

# ------------------ EVALUATION ------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ RESULTS ------------------
st.subheader("📊 Model Evaluation")

st.write(f"**Mean Squared Error (MSE):** `{mse:.2f}`")
st.write(f"**R² Score:** `{r2:.4f}`")

st.subheader("📐 Model Parameters")
st.write(f"**Coefficients:** {model.coef_}")
st.write(f"**Intercept:** {model.intercept_}")

st.divider()

# ------------------ USER INPUT ------------------
st.subheader("🔍 Predict Daily Temperature")

sunlight = st.number_input("Hours of Sunlight", min_value=0.0, max_value=24.0, value=6.0)
humidity = st.number_input("Humidity Level (%)", min_value=0.0, max_value=100.0, value=50.0)

if st.button("🌡️ Predict Temperature"):
    input_data = np.array([[sunlight, humidity]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Daily Temperature: **{prediction:.2f}°C**")

st.divider()

# ------------------ ACTUAL vs PREDICTED ------------------
st.subheader("📋 Sample Actual vs Predicted Values")

comparison = pd.DataFrame({
    "Actual Temperature": y_test.values[:5],
    "Predicted Temperature": y_pred[:5]
})

st.table(comparison)
