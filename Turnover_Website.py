# === IMPORT LIBRARY === #
import streamlit as st
import pandas as pd
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Turnover", layout="centered")

# Memanggil CSS eksternal (jika ada)
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# === MENU SIDEBAR ===
menu = st.sidebar.selectbox("Pilih Halaman", ["Home", "How To Use?", "About Us"])

# Load model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# === HOME PAGE ===
if menu == "Home":
    st.markdown('<div class="title-custom">🌿 Prediksi Turnover Karyawan</div>', unsafe_allow_html=True)
    st.caption("Masukkan data secara manual atau upload file CSV untuk memprediksi Turnover karyawan.")

    # Pilihan metode input
    input_method = st.radio("Pilih metode input data:", ["Input Manual", "Upload CSV"])

    if input_method == "Input Manual":
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
                st.error("⚠️ Hasil Prediksi: Karyawan **berpotensi keluar**.")
            else:
                st.success("✅ Hasil Prediksi: Karyawan **kemungkinan tetap tinggal**.")

    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Data yang diupload:")
            st.dataframe(df)

            # Pastikan format sesuai
            df_encoded = pd.get_dummies(df)
            for col in model.feature_names_in_:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[model.feature_names_in_]

            preds = model.predict(df_encoded)
            df["Prediksi Turnover"] = ["Keluar" if p == 1 else "Tetap" for p in preds]

            st.write("📊 Hasil Prediksi:")
            st.dataframe(df)

            # Download hasil
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Hasil Prediksi", csv, "hasil_prediksi.csv", "text/csv")

# === HOW TO USE? PAGE ===
elif menu == "How To Use?":
    st.markdown('<div class="title-custom">📘 Panduan Penggunaan</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Input Manual**: Isi form di halaman Home untuk memprediksi 1 karyawan.
    - **Upload CSV**: Unggah file CSV berisi data karyawan untuk memprediksi banyak data sekaligus.
    - Pastikan format kolom sesuai dengan fitur yang digunakan model.
    """)

# === ABOUT US PAGE ===
elif menu == "About Us":
    st.markdown('<div class="title-custom">Tentang Saya</div>', unsafe_allow_html=True)
    st.write("""
        Anda dapat menghubungi saya melalui:
        
        - 📧 Email: s32210049@student.ubm.ac.id  
        - ☎️ Telepon: +62 813-8424-5198  
        - 🌐 Instagram: @alexxpardedee
    """)

# === WATERMARK ===
st.sidebar.markdown("""<div class="sidebar-watermark">© 2025 Skripsi Alexandro T Pardede</div>""", unsafe_allow_html=True)
