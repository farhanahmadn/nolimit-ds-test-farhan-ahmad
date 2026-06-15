import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from PIL import Image

st.set_page_config(page_title="Deteksi Spam", page_icon="📧", layout="wide")
st.title("📧 Sistem Deteksi Spam Email Indonesia")
st.markdown("*Developed for NoLimit Indonesia Data Scientist Test*")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    classifier = joblib.load("spam_classifier_model.pkl")
    return embedder, classifier

embedder, classifier = load_models()

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Interaktif", "📊 Performa Model", "🏗️ Arsitektur Sistem"])

with tab1:
    st.subheader("Uji Model Klasifikasi")
    st.write("Masukkan teks email untuk mendeteksi apakah itu Spam atau Ham (Aman).")
    
    user_input = st.text_area("Teks Email:", placeholder="Ketik isi pesan di sini...", height=150)
    
    if st.button("Analisis Teks"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Mengekstrak fitur dan menganalisis..."):
                embedding = embedder.encode([user_input])
                prediction = classifier.predict(embedding)[0]
                
                st.markdown("---")
                if prediction.lower() == 'spam':
                    st.error(f"**Hasil Prediksi: SPAM** 🚨")
                else:
                    st.success(f"**Hasil Prediksi: HAM (Aman)** ✅")

with tab2:
    st.subheader("Hasil Evaluasi Training")
    st.write("Model dilatih menggunakan Scikit-Learn Logistic Regression di atas fitur numerik (384-Dimensi) dari Hugging Face `paraphrase-multilingual-MiniLM-L12-v2`.")
    
    # Hasil Metrik Evaluasi ini didapat dari proses training yang sudah dilakukan di main.py
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Keseluruhan", "97%")
    col2.metric("F1-Score (Spam)", "0.97")
    col3.metric("F1-Score (Ham)", "0.97")
    
    st.markdown("### Classification Report")
    st.code("""
              precision    recall  f1-score   support

         ham       0.99      0.95      0.97       263
        spam       0.95      0.99      0.97       265

    accuracy                           0.97       528
   macro avg       0.97      0.97      0.97       528
weighted avg       0.97      0.97      0.97       528
    """, language="text")

with tab3:
    st.subheader("End-to-End NLP Pipeline")
    st.write("Visualisasi aliran data dari teks mentah hingga probabilitas prediksi akhir.")
    
    try:
        # Pastikan gambar flowchart Anda bernama pipeline_flowchart.png
        image = Image.open('pipeline_flowchart.png')
        st.image(image, caption='Arsitektur Pemrosesan Data', use_column_width=True)
    except FileNotFoundError:
        st.warning("⚠️ File 'pipeline_flowchart.png' belum ditemukan di folder ini. Pastikan Anda sudah menyimpan gambar dari Mermaid sebelumnya!")

    st.markdown("---")
    st.subheader("💡 Justifikasi Arsitektur: Pipeline vs Fine-Tuning IndoBERT")
    
    st.info("""
    Dalam mendesain sistem ini, pendekatan *Feature Extraction* (Sentence-Transformers + Sklearn) dipilih secara sadar dibandingkan melakukan *full fine-tuning* pada model Language Model besar seperti **IndoBERT**. Berikut adalah dasar pertimbangannya:
    
    **1. Kepatuhan Spesifik pada Syarat Tes** Instruksi tes mewajibkan penggunaan *embeddings* sebagai representasi yang diumpankan ke algoritma *machine learning* standar (Sklearn/ANN). Melakukan *fine-tuning* IndoBERT secara *end-to-end* (menggunakan klasifikasi bawaan model) justru akan menyimpang dari instruksi arsitektur yang diminta.
    
    **2. Kualitas Representasi Level Kalimat (Sentence vs Word Embeddings)** IndoBERT murni berarsitektur berbasis token/kata. Tanpa mekanisme *pooling* tambahan yang kompleks, ia kurang optimal untuk merepresentasikan satu teks *email* utuh. Sebaliknya, model `paraphrase-multilingual-MiniLM-L12-v2` dari *library* Sentence-Transformers sudah dilatih secara khusus (*out-of-the-box*) untuk memampatkan makna seluruh kalimat/paragraf menjadi satu vektor padat (384-Dimensi) yang sangat berkualitas.

    **3. Efisiensi Resource & Deployment (Micro-Architecture)** *Fine-tuning* IndoBERT membutuhkan waktu berjam-jam, memori GPU yang besar, dan menghasilkan ukuran file model (bobot) ratusan Megabyte hingga Gigabyte yang sulit di-*deploy*. Dengan memisahkan proses (Hugging Face hanya sebagai pengekstrak fitur, Sklearn sebagai otak pengambil keputusan), kita mendapatkan sistem yang secepat kilat. Proses *training* lokal selesai dalam hitungan detik, dan ukuran model *classifier* (`.pkl`) sangat kecil, sehingga 100% aman dari risiko *Out-of-Memory* (OOM) saat di-*deploy* ke Streamlit Cloud atau *server* skala kecil.
    """)