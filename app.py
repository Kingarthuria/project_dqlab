import streamlit as st
import pandas as pd
import pickle 
import time
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# Open Pickle
location_file = 'generate_heart_disease.pkl'
with open(location_file, 'rb') as file:
    model = pickle.load(file)


# Page Configuration

st.set_page_config(
    page_title='Heart Disease Prediction',
    layout='wide'
)


add_selectitem = st.sidebar.selectbox("Want to open about?", ("About this app!", "Heart Disease!", "About Creator!"))

def heart():

    st.title('Prediksi Risiko Penyakit Jantung')
    
    st.markdown("""
    Aplikasi ini dirancang untuk membantu Anda melakukan **skrining mandiri** terhadap risiko penyakit jantung. 
    Kami menggunakan algoritma **Random Forest Classifier** yang telah dilatih menggunakan data medis historis untuk memberikan prediksi yang akurat.
    """)

    # Menambahkan Petunjuk Penggunaan dalam Expander agar rapi
    with st.expander("📖 Cara Menggunakan Aplikasi"):
        st.markdown("""
        1. **Isi Data Medis:** Masukkan informasi kesehatan Anda pada panel **Sidebar di sebelah kiri**.
        2. **Perhatikan Keterangan:** Baca keterangan di bawah slider/selectbox untuk memahami setiap parameter.
        3. **Klik Tombol Predict:** Setelah semua data terisi, tekan tombol **'Predict!'** di sidebar.
        4. **Lihat Hasil:** Hasil analisis akan muncul pada bagian bawah.
        """)

    st.divider()

    # Fungsi untuk menangkap input dari sidebar
    def user_input_features():
        st.sidebar.header('Manual Input Pasien')

        # =========================
        # Mapping Kategori (Indonesia → Numeric)
        # =========================
        sex_map = {
            'Perempuan': 0,
            'Laki-laki': 1
        }

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

        exang_map = {
            'Tidak': 0,
            'Ya': 1
        }

        thal_map = {
            'Normal': 1,
            'Cacat tetap (fixed defect)': 2,
            'Cacat reversibel (reversable defect)': 3
        }

        # =========================
        # Input User (Human-readable)
        # =========================
        sex_label = st.sidebar.selectbox("Jenis Kelamin", list(sex_map.keys()))
        cp_label = st.sidebar.selectbox("Tipe Nyeri Dada (CP)", list(cp_map.keys()))
        slope_label = st.sidebar.selectbox("Kemiringan Segmen ST (Slope)", list(slope_map.keys()))
        exang_label = st.sidebar.selectbox("Nyeri Dada saat Berolahraga (Exang)", list(exang_map.keys()))
        thal_label = st.sidebar.selectbox("Hasil Tes Thalium", list(thal_map.keys()))

        thalach = st.sidebar.slider("Detak Jantung Maksimum (thalach)", 88, 202, 150)
        oldpeak = st.sidebar.slider("Depresi ST (oldpeak)", 0.0, 4.0, 1.0)

        # ⬅️ CA tetap seperti sebelumnya (angka)
        ca = st.sidebar.slider(
            "Jumlah Pembuluh Darah Utama (CA)",
            min_value=0,
            max_value=3,
            value=0,
            help="Nilai 0–3 menunjukkan jumlah pembuluh darah utama"
        )

        age = st.sidebar.slider("Usia", 29, 77, 45)

        # =========================
        # Konversi ke Numeric (untuk Model)
        # =========================
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

        features = pd.DataFrame(data, index=[0])
        return features


    # Menjalankan fungsi input
    input_df = user_input_features()

    st.subheader('Data Pasien yang Akan Diprediksi:')
    st.write(input_df)

    # Tombol Prediksi
    if st.sidebar.button('Predict!'):
        with st.spinner('Menganalisis data... Mohon tunggu'):
            time.sleep(2) # Durasi animasi 2 detik (lebih nyaman untuk user)
            
            # Prediksi menggunakan model yang sudah di-load di awal script
            prediction = model.predict(input_df)
            
            st.divider()
            st.subheader('Hasil Analisis Model: ')
            
            if prediction[0] == 0:
                st.success("Hasil: **No Heart Disease** (Kondisi jantung terdeteksi normal)")
            else:
                st.error("Hasil: **Heart Disease** (Terdeteksi risiko penyakit jantung)")
            
            st.caption("Selalu konsultasikan hasil ini dengan tenaga medis profesional.")


def about_heart_disease():

    st.title("Mengenal Penyakit Jantung")
    
    st.write("""
    Penyakit jantung adalah kondisi di mana jantung mengalami gangguan, baik pada pembuluh darah jantung, 
    katup jantung, atau otot jantung. Berdasarkan data WHO, penyakit kardiovaskular adalah penyebab 
    kematian nomor satu di dunia.
    """)
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚩 Faktor Risiko Utama")
        st.write("""
        * **Tekanan Darah Tinggi (Hipertensi):** Beban kerja jantung meningkat.
        * **Kolesterol Tinggi:** Penumpukan plak di pembuluh darah.
        * **Gaya Hidup:** Merokok, kurang olahraga, dan pola makan buruk.
        * **Diabetes:** Gula darah tinggi merusak pembuluh darah.
        """)

    with col2:
        st.subheader("🩺 Gejala Umum")
        st.write("""
        * Nyeri dada (Angina).
        * Sesak napas saat beraktivitas.
        * Detak jantung tidak teratur (Palpitasi).
        * Kelelahan ekstrem tanpa sebab jelas.
        """)

    st.write("---") # Garis tipis tambahan
    st.markdown(f"""
    ### 💡 Mengapa Aplikasi Ini Diciptakan?
    Melihat tingginya angka risiko penyakit jantung dan kompleksitas gejala yang ada, 
    **kesadaran dini adalah kunci utama**. Oleh karena itu, saya mengembangkan aplikasi 
    **Heart Disease Prediction** ini sebagai alat bantu skrining awal secara mandiri.
    
    Dengan memanfaatkan teknologi *Machine Learning*, aplikasi ini diharapkan dapat memberikan 
    gambaran cepat mengenai kondisi kesehatan jantung Anda berdasarkan parameter medis yang 
    umum digunakan oleh para ahli. Seperti yang sering dikatakan, mencegah selalu lebih baik 
    daripada mengobati—dan **pahlawan Himmel pun pasti akan setuju bahwa membantu orang lain 
    menjaga kesehatannya adalah hal yang tepat untuk dilakukan.**
    """)

    st.divider()


    st.subheader("Penjelasan Fitur dalam Model")
    with st.expander("Klik untuk melihat penjelasan teknis variabel"):
        st.write("""
        Model Machine Learning ini memproses data medis Anda berdasarkan beberapa fitur berikut:
        1. **Chest Pain Type (CP):** Tingkat keparahan nyeri dada yang dirasakan.
        2. **Thalach:** Detak jantung maksimal yang dicapai saat tes beban.
        3. **Oldpeak:** Depresi segmen ST yang menunjukkan adanya sumbatan aliran darah ke jantung.
        4. **CA:** Jumlah pembuluh darah utama (0-3) yang terlihat melalui fluoroskopi.
        5. **Thal:** Hasil tes darah thalium (Normal, Cacat tetap, atau Cacat reversibel).
        """)

    st.info("""
    **Catatan Penting:** Aplikasi ini adalah alat bantu berbasis *Machine Learning* dan bukan merupakan saran medis profesional. 
    Selalu konsultasikan hasil kesehatan Anda dengan dokter spesialis jantung (kardiolog).
    """)

def about_me():
    st.title("👤 About the Developer")
    st.write("##")
    
    col1, col2 = st.columns([1, 2])
    
    with col1: 
        st.image("profile_image.jpg", width=200) # Aktifkan ini jika sudah ada foto

    with col2:
        st.subheader("Zacky Bayu Prasongko")
        st.write("""
        **Mahasiswa Sistem Informasi**
        
        Halo! Saya Zacky, seorang pembelajar di bidang Teknologi Informasi yang sedang mendalami dunia *Data Science* dan *Machine Learning*. 
        Saat ini, saya sedang fokus mengembangkan kemampuan dalam mengolah data menjadi informasi yang bermanfaat melalui berbagai proyek praktis.
        """)

    st.divider()

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🚀 Project Background")
        st.write("""
        Aplikasi **Heart Disease Prediction** ini merupakan proyek akhir saya dalam **Machine Learning Bootcamp di DQLAB**. 
        Proyek ini mencakup:
        * Data Cleaning & Exploration.
        * Model Building menggunakan Random Forest.
        * Deployment menggunakan Streamlit.
        """)

    with col_b:
        st.subheader("🛠️ My Journey")
        st.write("""
        Sebagai seorang pemula, saya percaya bahwa konsistensi adalah kunci. 
        Keahlian yang sedang saya asah:
        * Python Programming.
        * Data Analysis .
        * Machine Learning Workflow.
        """)

    st.divider()

    st.info(f"✨ **Favorite Quote:** \"Karena pahlawan Himmel pasti akan melakukan hal tersebut.\"")
    
    st.write("---")
    st.write("### Let's Connect!")
    col_ig, col_li, col_mail = st.columns(3)
    
    with col_ig:
        st.write("📷 **Instagram**")
        st.write("[@nero_oid](https://www.instagram.com/nero_oid/)")
        
    with col_li:
        st.write("💼 **LinkedIn**")
        # Ganti 'zacky-bayu' dengan username LinkedIn kamu yang asli
        st.write("[Let's Connect](https://www.linkedin.com/in/zackybayup/)")
        
    with col_mail:
        st.write("📧 **Email**")
        # Ganti dengan alamat Gmail kamu
        st.write("thedumbestknightever@gmail.com")
    


if add_selectitem == "About this app!":
    # Tambahkan judul dashboard di sini agar muncul sekali di atas
    st.title('Heart Disease Prediction App')
    st.write("### Welcome to my machine learning dashboard")
    st.write("Created by: [@nero_oid](https://www.instagram.com/nero_oid/)")
    about_heart_disease()
elif add_selectitem == "Heart Disease!":
    heart()
elif add_selectitem == "About Creator!":
    about_me()



