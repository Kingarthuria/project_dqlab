# Heart Disease Prediction

Aplikasi skrining mandiri risiko penyakit jantung berbasis Machine Learning, dibangun menggunakan Streamlit. Proyek ini merupakan tugas akhir dari Machine Learning Bootcamp di DQLAB.

## Tentang Proyek

Penyakit kardiovaskular adalah penyebab kematian nomor satu secara global, dengan gejala yang sering muncul terlambat. Proyek ini mengembangkan model klasifikasi untuk memprediksi risiko penyakit jantung berdasarkan parameter medis, sebagai alat bantu screening awal sebelum pemeriksaan klinis lebih lanjut.

## Dataset

Dataset yang digunakan adalah Heart Disease dari UCI Machine Learning Repository, terdiri dari 1025 records dengan 13 attribute dan 1 target variable.

Sumber: https://archive.ics.uci.edu/dataset/45/heart+disease

## Metodologi

1. Data Cleaning: penanganan tipe data, imputasi modus untuk nilai yang salah, pengecekan null dan duplikat, identifikasi dan penanganan outlier.
2. Exploratory Data Analysis: analisis distribusi target, statistik deskriptif, dan korelasi antar variabel.
3. Feature Selection: dari 13 fitur awal, dipilih 9 fitur dengan korelasi kuat terhadap target, yaitu cp, thalach, slope, oldpeak, exang, ca, thal, sex, dan age.
4. Modeling: perbandingan empat algoritma, Logistic Regression, Decision Tree, Random Forest, dan Multi-Layer Perceptron, dengan proses hyperparameter tuning.
5. Threshold Optimization: penentuan titik potong optimal menggunakan Youden's Index.

## Model Terpilih

Random Forest Classifier dipilih sebagai model final dengan hasil evaluasi sebagai berikut.

| Model | Accuracy | ROC-AUC | Threshold |
|---|---|---|---|
| Logistic Regression | 0.84 | 0.88 | 0.427 |
| Random Forest | 0.86 | 0.90 | 0.53 |
| Decision Tree | 0.79 | 0.84 | 0.608 |
| MLP | 0.82 | 0.88 | 0.587 |

Random Forest dipilih karena memiliki akurasi dan ROC-AUC tertinggi, serta lebih tahan terhadap variasi data dibandingkan Decision Tree tunggal.

## Fitur Aplikasi

Aplikasi memiliki tiga halaman.

- About this app, berisi penjelasan umum tentang penyakit jantung dan cara kerja model.
- Heart Disease Prediction, halaman input data pasien dan hasil prediksi.
- About Creator, informasi tentang pengembang proyek.

## Cara Menjalankan Secara Lokal

Clone repository ini, lalu install dependency yang dibutuhkan.

```bash
pip install streamlit pandas scikit-learn
```

Jalankan aplikasi dengan perintah berikut.

```bash
streamlit run app.py
```

Pastikan file `generate_heart_disease.pkl` berada pada direktori yang sama dengan `app.py`.

## Live Demo

Aplikasi dapat dicoba langsung melalui tautan berikut.

https://projectdqlabheartdisease.streamlit.app/

## Disclaimer

Aplikasi ini adalah alat bantu berbasis Machine Learning dan bukan merupakan saran medis profesional. Selalu konsultasikan hasil kesehatan Anda dengan dokter spesialis jantung.

## Kontak

Zacky Bayu Prasongko

- Instagram: [@nero_oid](https://www.instagram.com/nero_oid/)
- LinkedIn: [zackybayup](https://www.linkedin.com/in/zackybayup/)
- Email: thedumbestknightever@gmail.com
