import streamlit as st
import pandas as pd
import pickle
import time

# ============================================================
# Konfigurasi Model
# ============================================================
MODEL_PATH = 'generate_heart_disease.pkl'
THRESHOLD = 0.53  # Sesuai hasil Youden's Index (lihat laporan slide 29-30)

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title='Heart Disease Prediction',
    layout='wide'
)

add_selectitem = st.sidebar.selectbox(
    "Want to open about?",
    ("About this app!", "Heart Disease Prediction!", "About Creator!")
)


def heart():
    st.title('Prediksi Risiko Penyakit Jantung')

    st.markdown("""
    Aplikasi ini membantu Anda melakukan skrining mandiri terhadap risiko penyakit jantung
    menggunakan model Random Forest Classifier yang dilatih dari data medis historis.
    """)

    with st.expander("Cara Menggunakan Aplikasi"):
        st.markdown("""
        1. Isi data medis Anda pada panel sidebar di sebelah kiri.
        2. Baca keterangan di bawah slider/selectbox untuk memahami tiap parameter.
        3. Tekan tombol 'Predict!' di sidebar setelah semua data terisi.
        4. Hasil analisis akan muncul di bagian bawah.
        """)

    st.info(
        "Hasil aplikasi ini adalah alat bantu skrining berbasis Machine Learning, "
        "bukan diagnosis medis resmi. Konsultasikan hasil ini dengan dokter spesialis jantung (kardiolog)."
    )

    st.divider()

    def user_input_features():
        st.sidebar.header('Manual Input Pasien')

        sex_map = {'Perempuan': 0, 'Laki-laki': 1}
        cp_map = {
            'Angina tipikal': 0,
            'Angina atipikal': 1,
            'Nyeri non-angina': 2,
            'Tanpa gejala (asimtomatik)': 3
        }
        slope_map = {
            'Menurun (downsloping)': 0,
            'Datar (flat)': 1,
            'Meningkat (upsloping)': 2
        }
        exang_map = {'Tidak': 0, 'Ya': 1}
        thal_map = {
            'Normal': 1,
            'Cacat tetap (fixed defect)': 2,
            'Cacat reversibel (reversable defect)': 3
        }

        with st.sidebar.container(border=True):
            st.caption("PROFIL PASIEN")
            sex_label = st.selectbox("Jenis Kelamin", list(sex_map.keys()))
            age = st.slider("Usia", 29, 77, 45)

        with st.sidebar.container(border=True):
            st.caption("HASIL UJI KLINIS")
            cp_label = st.selectbox("Tipe Nyeri Dada (CP)", list(cp_map.keys()))
            thalach = st.slider("Detak Jantung Maksimum (thalach)", 88, 202, 150)
            exang_label = st.selectbox("Nyeri Dada saat Berolahraga (Exang)", list(exang_map.keys()))
            oldpeak = st.slider("Depresi ST (oldpeak)", 0.0, 4.0, 1.0)
            slope_label = st.selectbox("Kemiringan Segmen ST (Slope)", list(slope_map.keys()))
            ca = st.slider(
                "Jumlah Pembuluh Darah Utama (CA)",
                min_value=0, max_value=3, value=0,
                help="Nilai 0–3 menunjukkan jumlah pembuluh darah utama"
            )
            thal_label = st.selectbox("Hasil Tes Thalium", list(thal_map.keys()))

        data = {
            'cp': cp_map[cp_label],
            'thalach': thalach,
            'slope': slope_map[slope_label],
            'oldpeak': oldpeak,
            'exang': exang_map[exang_label],
            'ca': ca,
            'thal': thal_map[thal_label],
            'sex': sex_map[sex_label],
            'age': age
        }

        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    st.subheader('Data Pasien yang Akan Diprediksi')
    st.dataframe(input_df, hide_index=True)

    if st.sidebar.button('Predict!'):
        with st.spinner('Menganalisis data...'):
            time.sleep(1)

            probability = model.predict_proba(input_df)[0][1]
            prediction = 1 if probability >= THRESHOLD else 0

            st.divider()
            st.subheader('Hasil Analisis Model')

            col1, col2 = st.columns([2, 1])

            with col1:
                if prediction == 1:
                    st.error("Heart Disease — Terdeteksi risiko penyakit jantung")
                else:
                    st.success("No Heart Disease — Kondisi jantung terdeteksi normal")

                st.write(f"Probabilitas risiko: {probability:.1%}")

                if probability >= 0.75:
                    bar_color = "#e02424"  # merah tua — bahaya, segera periksa
                elif probability >= THRESHOLD:
                    bar_color = "#f59e0b"  # oranye — berisiko
                else:
                    bar_color = "#16a34a"  # hijau — normal

                st.markdown(
                    f"""
                    <div style="background-color:#333; border-radius:6px; height:14px; width:100%;">
                        <div style="background-color:{bar_color}; width:{probability*100}%;
                                    height:100%; border-radius:6px;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if probability >= 0.75:
                    st.caption("Skor risiko sangat tinggi — segera periksakan diri ke dokter.")

            with col2:
                st.metric("Skor Risiko", f"{probability:.1%}")


def about_heart_disease():
    st.title("Mengenal Heart Disease / Penyakit Jantung")

    st.write("""
    Penyakit jantung adalah kondisi di mana jantung mengalami gangguan, baik pada pembuluh darah jantung,
    katup jantung, atau otot jantung. Berdasarkan data WHO, penyakit kardiovaskular adalah penyebab
    kematian nomor satu di dunia.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Faktor Risiko Utama")
        st.write("""
        - Tekanan darah tinggi (hipertensi), yang meningkatkan beban kerja jantung.
        - Kolesterol tinggi, yang memicu penumpukan plak di pembuluh darah.
        - Gaya hidup: merokok, kurang olahraga, pola makan buruk.
        - Diabetes, karena gula darah tinggi merusak pembuluh darah.
        """)

    with col2:
        st.subheader("Gejala Umum")
        st.write("""
        - Nyeri dada (angina).
        - Sesak napas saat beraktivitas.
        - Detak jantung tidak teratur (palpitasi).
        - Kelelahan ekstrem tanpa sebab jelas.
        """)

    st.divider()
    st.markdown("""
    ### Mengapa Aplikasi Ini Dibuat
    Gejala penyakit jantung sering muncul terlambat, sehingga kesadaran dan deteksi dini menjadi penting.
    Aplikasi ini dibangun sebagai alat bantu skrining awal secara mandiri, memanfaatkan model Machine Learning
    untuk memberi gambaran cepat mengenai kondisi kesehatan jantung berdasarkan parameter medis yang umum digunakan.
    """)

    st.divider()

    st.subheader("Penjelasan Fitur dalam Model")
    with st.expander("Klik untuk melihat penjelasan teknis variabel"):
        st.write("""
        Model ini memproses data pasien berdasarkan sembilan fitur yang terpilih melalui tahap feature
        selection karena korelasinya yang kuat terhadap target:

        - CP (Chest Pain Type): tingkat keparahan nyeri dada.
        - Thalach: detak jantung maksimal saat tes beban.
        - Slope: kemiringan segmen ST saat fisik maksimal.
        - Oldpeak: depresi segmen ST yang mengindikasikan sumbatan aliran darah.
        - Exang: nyeri dada akibat olahraga.
        - CA: jumlah pembuluh darah utama (0-3) yang terlihat melalui fluoroskopi.
        - Thal: hasil tes darah thalium.
        - Sex: jenis kelamin pasien.
        - Age: usia pasien.
        """)

    st.info(
        "Aplikasi ini adalah alat bantu berbasis Machine Learning dan bukan merupakan saran medis profesional. "
        "Konsultasikan hasil kesehatan Anda dengan dokter spesialis jantung (kardiolog)."
    )


def about_me():
    st.title("About the Developer")

    col1, col2 = st.columns([1, 2])

    with col1:
        try:
            st.image("profile_image.jpg", width=200)
        except Exception:
            pass

    with col2:
        st.subheader("Zacky Bayu Prasongko")
        st.write("""
        Mahasiswa Sistem Informasi, sedang mendalami Data Science dan Machine Learning.
        Saat ini fokus mengembangkan kemampuan mengolah data menjadi informasi yang bermanfaat
        melalui proyek-proyek praktis.
        """)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Tentang Proyek Ini")
        st.write("""
        Aplikasi Heart Disease Prediction ini adalah proyek akhir dari Machine Learning Bootcamp di DQLAB,
        mencakup data cleaning dan eksplorasi, pembangunan model dengan Random Forest,
        dan deployment menggunakan Streamlit.
        """)

    with col_b:
        st.subheader("Yang Sedang Dipelajari")
        st.write("""
        - Python programming.
        - Data analysis.
        - Machine learning workflow.
        """)

    st.divider()

    st.write("### Kontak")
    col_ig, col_li, col_mail = st.columns(3)

    with col_ig:
        st.write("Instagram")
        st.write("[@nero_oid](https://www.instagram.com/nero_oid/)")

    with col_li:
        st.write("LinkedIn")
        st.write("[Arthur Pendragon](https://www.linkedin.com/in/arthurpendragon/)")

    with col_mail:
        st.write("Email")
        st.write("thedumbestknightever@gmail.com")


if add_selectitem == "About this app!":
    st.title('Heart Disease Prediction App')
    st.write("Dashboard machine learning untuk skrining risiko penyakit jantung.")
    st.write("Created by: [@nero_oid](https://www.instagram.com/nero_oid/)")
    about_heart_disease()
elif add_selectitem == "Heart Disease Prediction!":
    heart()
elif add_selectitem == "About Creator!":
    about_me()