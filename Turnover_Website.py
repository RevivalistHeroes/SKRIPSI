import streamlit as st
import pandas as pd
import pickle

# Konfigurasi halaman (tanpa sidebar)
st.set_page_config(page_title="Prediksi Turnover", layout="centered")

# Gaya minimalis
st.markdown("""
    <style>
    .stApp {
        font-family: 'Arial', sans-serif;
        background-color: #f9f9f9;
        color: #333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5em 1em;
        margin-top: 1em;
    }
    .stSlider > div {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Judul dan deskripsi
st.title("🔍 Prediksi Turnover Karyawan")
st.caption("Masukkan data karyawan untuk memprediksi apakah ia akan keluar dari perusahaan.")

# Load model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Form input
with st.form("form_prediksi"):
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
    last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5, step=0.01)
    number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
    time_spend_company = st.number_input("Years at Company", min_v
