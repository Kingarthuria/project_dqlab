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
    # Bagian ini fokus ke deskripsi singkat aplikasi
    st.write("""
    Aplikasi ini memprediksi risiko **Penyakit Jantung** menggunakan model Machine Learning (Random Forest). 
    Silakan masukkan data medis melalui panel di sebelah kiri.
    """)
    
    # Menampilkan Gambar dengan proteksi error (Try-Except)
    try:
        img = Image.open("heart-disease.jpg")
    
        col1, col2, col3 = st.columns([1, 2, 1])
    
        with col2:
            st.image(img, width=500)
        
    except FileNotFoundError:
        st.info("⚠️ Gambar Sedang Error / Tidak Tersedia")

    # Fungsi untuk menangkap input dari sidebar
    def user_input_features():
        st.sidebar.header('Manual Input Pasien')
        
        # Slider dan Input
        cp = st.sidebar.slider('Tipe Nyeri Dada (CP)', 1, 4, 2)
        # Menampilkan keterangan teks berdasarkan angka CP
        if cp == 1.0: wcp = "Nyeri dada tipe angina"
        elif cp == 2.0: wcp = "Nyeri dada tipe tidak stabil"
        elif cp == 3.0: wcp = "Nyeri dada tidak stabil parah"
        else: wcp = "Nyeri dada bukan masalah jantung"
        st.sidebar.caption(f"Keterangan: {wcp}")

        thalach = st.sidebar.slider("Detak Jantung Maksimum (thalach)", 88, 202, 150)
        slope = st.sidebar.slider("Kemiringan Segmen ST (slope)", 0, 2, 1)
        oldpeak = st.sidebar.slider("Depresi ST (oldpeak)", 0.0, 4.0, 1.0)
        exang_raw = st.sidebar.selectbox("Apakah Anda merasa nyeri dada saat berolahraga?", ("Tidak", "Ya"))
        exang = 1 if exang_raw == "Ya" else 0
        ca = st.sidebar.slider("Jumlah Pembuluh Darah Utama (ca)", 0, 3, 0)
        thal = st.sidebar.slider("Hasil Tes Thalium", 1, 3, 2)
        
        sex_raw = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
        sex = 1 if sex_raw == 'Pria' else 0
        
        age = st.sidebar.slider("Usia", 29, 77, 45)

        # Menyusun data ke DataFrame
        data = {'cp': cp, 'thalach': thalach, 'slope': slope, 'oldpeak': oldpeak,
                'exang': exang, 'ca': ca, 'thal': thal, 'sex': sex, 'age': age}
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
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("##") 
        st.title("👨‍💻") 
        # st.image("foto_zacky.jpg", width=200) # Aktifkan ini jika sudah ada foto

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
    st.write("📷 Instagram: [@nero_oid](https://www.instagram.com/nero_oid/)")


if add_selectitem == "About this app!":
    st.title('Heart Disease Prediction App')
    st.write("### Welcome to my machine learning dashboard")
    st.write("Created by: [@nero_oid](https://www.instagram.com/nero_oid/)")
    about_heart_disease()
elif add_selectitem == "Heart Disease!":
    heart()
elif add_selectitem == "About Creator!":
    about_me()



