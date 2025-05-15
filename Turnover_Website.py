import streamlit as st
import pandas as pd
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Turnover", layout="centered")

# Styling warna-warni dan menarik
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #2d2d2d;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    .stSlider > div {
        color: #333;
    }
    .stSelectbox label, .stNumberInput label {
        color: #222;
        font-weight: 500;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Judul dan deskripsi
st.title("🌈 Prediksi Turnover Karyawan")
st.caption("Masukkan data karyawan untuk memprediksi apakah mereka akan keluar dari perusahaan.")

# Load model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Formulir input
with st.form("form_prediksi"):
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
    last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5, step=0.01)
    number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
    time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=3)
    Work_accident = st.selectbox("Work Accident", [0, 1])
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years", [0, 1])
    sales = st.selectbox("Department", [
        "sales", "accounting", "hr", "technical", "support", 
        "management", "IT", "product_mng", "marketing", "RandD"
    ])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

    submit = st.form_submit_button("🔍 Prediksi Sekarang")

# Proses prediksi
if submit:
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

    if hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_]

    pred = model.predict(input_encoded)[0]

    if pred == 1:
        st.error("🚨 Hasil Prediksi: Karyawan **berpotensi keluar** dari perusahaan.")
    else:
        st.success("🎉 Hasil Prediksi: Karyawan **kemungkinan besar akan tetap tinggal**.")
