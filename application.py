import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load the model
model = joblib.load("xgb_pjm_model.pkl")

# Load data for visual trends (optional enhancement)
try:
    data = pd.read_excel("PJMW_MW_Hourly(1016).xlsx")
    data.columns = ["Datetime", "Load"]
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    data = data.sort_values("Datetime").reset_index(drop=True)
    data_available = True
except:
    data_available = False

# Page config
st.set_page_config(page_title="PJM Energy Forecast", layout="centered", page_icon="⚡")

# 🌗 Theme picker
theme = st.sidebar.selectbox("🎨 Select Theme", ["Day", "Night", "Solarized", "Monokai"])

# Theme CSS
if theme == "Day":
    st.markdown("""
        <style>
        .stApp { background-color: #f0f2f6; color: #000000; }
        </style>
    """, unsafe_allow_html=True)
elif theme == "Night":
    st.markdown("""
        <style>
        .stApp { background-color: #1e1e1e; color: #ffffff; }
        div[data-testid="stMarkdownContainer"] > *, .stMarkdown, .stRadio, .stSelectbox, .stSlider, .stNumberInput {
            color: #ffffff !important;
        }
        .stButton>button { background-color: #444444; color: white; }
        .stMetricLabel, .stMetricValue { color: #ffffff !important; }
        </style>
    """, unsafe_allow_html=True)
elif theme == "Solarized":
    st.markdown("""
        <style>
        .stApp { background-color: #fdf6e3; color: #586e75; }
        </style>
    """, unsafe_allow_html=True)
elif theme == "Monokai":
    st.markdown("""
        <style>
        .stApp { background-color: #272822; color: #f8f8f2; }
        div[data-testid="stMarkdownContainer"] > *, .stMarkdown, .stRadio, .stSelectbox, .stSlider, .stNumberInput {
            color: #f8f8f2 !important;
        }
        .stButton>button { background-color: #444444; color: white; }
        .stMetricLabel, .stMetricValue { color: #f8f8f2 !important; }
        </style>
    """, unsafe_allow_html=True)

# Title with icon
st.markdown("<h1 style='text-align: center;'>⚡ PJM Hourly Energy Forecast</h1>", unsafe_allow_html=True)
st.markdown("### Predict future energy consumption based on time and past load features.")

# Sidebar for inputs
st.sidebar.header("🔧 Input Features")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.number_input("Day of Month", 1, 31, 15)
month = st.sidebar.selectbox("Month", list(range(1, 13)))

# Input preview
st.markdown("#### Current Inputs:")
col1, col2, col3 = st.columns(3)
col1.metric("Hour", hour)
col2.metric("Day", day)
col3.metric("Month", month)

# Placeholder values (or automate if needed later)
lag1 = 5000.0
rolling_mean_24 = 5000.0

# Prediction
if st.button("🔍 Forecast"):
    features = pd.DataFrame([[hour, day, month, lag1, rolling_mean_24]],
                            columns=['Hour', 'Day', 'Month', 'lag1', 'rolling_mean_24'])
    forecast = model.predict(features)[0]

    # Suggestive forecast comment
    if forecast < 4000:
        comment = "⚠️ Low load – potentially off-peak hours."
    elif 4000 <= forecast < 7000:
        comment = "✅ Normal load – stable consumption."
    elif 7000 <= forecast < 10000:
        comment = "⚠️ High demand – monitor usage."
    else:
        comment = "🔥 Very high load – possible peak hour or risk of overload!"

    st.success(f"⚡ Predicted Load: **{forecast:.2f} MW**")
    st.info(comment)

    # Optional: Show 24-hour trend if data is available
    if data_available:
        try:
            now = datetime(datetime.now().year, month, day, hour)
            past_24 = data[data["Datetime"] < now].tail(24)
            st.markdown("#### 🔄 Load Trend (Past 24 Hours)")
            st.line_chart(past_24.set_index("Datetime")["Load"])
        except:
            st.warning("Unable to generate chart – invalid date/time range.")

# Footer
st.markdown("---")
st.markdown("<small>Developed for PJM Energy Data Analysis. Powered by XGBoost & Streamlit.</small>", unsafe_allow_html=True)
