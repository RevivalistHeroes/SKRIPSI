import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Prediksi Turnover", layout="wide")

# Gaya Kustom
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #003366;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stSlider > div {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=100)
    st.markdown("## Navigasi")
    st.markdown("- 🏠 Home\n- 📚 Tentang\n- 📈 Dokumentasi")
    st.markdown("---")
    st.markdown("### 📊 Tentang Aplikasi")
    st.info("Aplikasi ini memprediksi kemungkinan seorang karyawan keluar dari perusahaan berdasarkan data historis menggunakan model machine learning.")

# Judul Halaman
st.title("💼 Prediksi Turnover Karyawan")
st.write("Gunakan formulir di bawah untuk memasukkan data dan mendapatkan prediksi.")

# Load model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.subheader("📝 Formulir Input Manual untuk Prediksi")

with st.form("manual_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
        last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5, step=0.01)
        number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)

    with col2:
        time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=3)
        Work_accident = st.selectbox("Work Accident", [0, 1])
        promotion_last_5years = st.selectbox("Promoted in Last 5 Years", [0, 1])
        sales = st.selectbox("Department", [
            "sales", "accounting", "hr", "technical", "support", 
            "management", "IT", "product_mng", "marketing", "RandD"
        ])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
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
        st.error("🚨 Hasil Prediksi: **Keluar (1)** – Karyawan kemungkinan akan keluar.")
    else:
        st.success("✅ Hasil Prediksi: **Tidak Keluar (0)** – Karyawan kemungkinan akan tetap tinggal.")

    # (Opsional) Tambahkan visualisasi hasil dummy
    st.subheader("📊 Contoh Statistik Umum (Dummy)")
    fig, ax = plt.subplots()
    ax.bar(["Keluar", "Tidak Keluar"], [123, 456], color=["red", "green"])
    ax.set_ylabel("Jumlah Karyawan")
    st.pyplot(fig)
