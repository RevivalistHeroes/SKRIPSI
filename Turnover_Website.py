import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Prediksi", layout="wide")
st.title("Prediksi Otomatis dari Dataset")


# Load model statis dari file
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    pickle.dump(MODEL_PATH, model)

st.subheader("Formulir Input Manual untuk Prediksi")

with st.form("manual_input_form"):
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
    last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5, step=0.01)
    number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
    time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=3)
    Work_accident = st.selectbox("Work Accident", [0, 1])
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years", [0, 1])
    sales = st.selectbox("Department", [
        "sales", "accounting", "hr", "technical", "support", 
        "management", "IT", "product_mng", "marketing", 
        "RandD"
    ])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Buat dataframe satu baris dari input
    input_dict = {
        'satisfaction_level': [satisfaction_level],
        'last_evaluation': [last_evaluation],
        'number_project': [number_project],
        'average_montly_hours': [average_montly_hours],
        'time_spend_company': [time_spend_company],
        'Work_accident': [Work_accident],
        'promotion_last_5years': [promotion_last_5years],
        'sales': [sales],
        'salary': [salary],
    }

    input_df = pd.DataFrame(input_dict)
    input_encoded = pd.get_dummies(input_df)

    # Tambahkan kolom yang hilang agar cocok dengan model
    if hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_]

    # Prediksi
    pred = model.predict(input_encoded)[0]
    st.success(f"Hasil Prediksi: {'Keluar (1)' if pred == 1 else 'Tidak Keluar (0)'}")
