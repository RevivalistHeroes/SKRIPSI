# === IMPORT LIBRARY === #
import streamlit as st
import pandas as pd
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Turnover", layout="centered")

# Memanggil CSS eksternal
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Navigasi Sidebar
menu = st.sidebar.selectbox("Pilih Halaman", ["Home", "How To Use?", "About Us"])

# === HOME PAGE ===
if menu == "Home":
    st.markdown('<div class="title-custom">📌 Prediksi Turnover Karyawan</div>', unsafe_allow_html=True)
    st.caption("Masukkan data untuk mengetahui kemungkinan seorang karyawan keluar dari perusahaan.")

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
            st.error("⚠️ Hasil Prediksi: Karyawan **berpotensi keluar**.")
        else:
            st.success("✅ Hasil Prediksi: Karyawan **kemungkinan tetap tinggal**.")

# === HOW TO USE? PAGE ===
elif menu == "How To Use?":
    st.markdown('<div class="title-custom">📘 Panduan Penggunaan</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>1. Satisfaction Level</h4>
        <p>Tingkat kepuasan karyawan terhadap pekerjaannya. Nilai antara <strong>0.0 - 1.0</strong>, di mana:
        <br> - 1.0 = sangat puas
        <br> - 0.0 = sangat tidak puas</p>
    </div>

    <div class="info-box">
        <h4>2. Last Evaluation</h4>
        <p>Nilai evaluasi terakhir dari karyawan. Skala <strong>0.0 - 1.0</strong>, mencerminkan kinerja terakhir.</p>
    </div>

    <div class="info-box">
        <h4>3. Number of Projects</h4>
        <p>Jumlah proyek yang sedang dikerjakan oleh karyawan. Nilai terlalu rendah/tinggi bisa menunjukkan underload atau overload.</p>
    </div>

    <div class="info-box">
        <h4>4. Average Monthly Hours</h4>
        <p>Rata-rata jam kerja per bulan. Digunakan untuk melihat beban kerja karyawan.</p>
    </div>

    <div class="info-box">
        <h4>5. Years at Company</h4>
        <p>Berapa lama karyawan sudah bekerja di perusahaan.</p>
    </div>

    <div class="info-box">
        <h4>6. Work Accident</h4>
        <p>Apakah karyawan pernah mengalami kecelakaan kerja:
        <br> - 1 = Ya
        <br> - 0 = Tidak</p>
    </div>

    <div class="info-box">
        <h4>7. Promoted in Last 5 Years</h4>
        <p>Apakah karyawan mendapat promosi dalam 5 tahun terakhir:
        <br> - 1 = Ya
        <br> - 0 = Tidak</p>
    </div>

    <div class="info-box">
        <h4>8. Department</h4>
        <p>Departemen tempat karyawan bekerja, seperti sales, IT, marketing, dll.</p>
    </div>

    <div class="info-box">
        <h4>9. Salary Level</h4>
        <p>Tingkat gaji karyawan:
        <br> - low = rendah
        <br> - medium = sedang
        <br> - high = tinggi</p>
    </div>

    <div class="info-box">
        <h4>🔍 Prediksi</h4>
        <p>Setelah semua kolom diisi, klik tombol <strong>Prediksi Sekarang</strong> untuk melihat hasil apakah karyawan akan keluar atau tidak.</p>
    </div>
    """, unsafe_allow_html=True)

# === ABOUT US PAGE ===
elif menu == "About Us":
    st.markdown('<div class="title-custom">Tentang Saya</div>', unsafe_allow_html=True)
    st.write("""
        Anda dapat menghubungi saya melalui:
        
        - 📧 Email: s32210049@student.ubm.ac.id  
        - ☎️ Telepon: +62 813-8424-5198  
        - 🌐 Instagram: @alexxpardedee
        
        Atau isi form berikut untuk pertanyaan:
    """)
    with st.form("form_kontak"):
        nama = st.text_input("Nama")
        email = st.text_input("Email")
        pesan = st.text_area("Pesan")
        kirim = st.form_submit_button("Kirim")

    if kirim:
        st.success(f"Terima kasih, {nama}! Pesan Anda sudah kami terima.")


# === WATERMARK ===
st.sidebar.markdown("""
    <div class="sidebar-watermark">© 2025 Skripsi Alexandro T Pardede</div>
""", unsafe_allow_html=True)
