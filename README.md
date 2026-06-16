# Indonesia Email Spam Detection System 

**Repository Name:** `nolimit-ds-test-farhan-ahmad` 
**Task Option:** A (Classification - Spam/Ham Detection for Indonesian Emails)

Proyek ini dibangun untuk memenuhi kualifikasi tes Data Scientist di NoLimit Indonesia. Sistem ini merupakan sebuah *end-to-end pipeline Machine Learning* untuk mengklasifikasikan teks *email* berbahasa Indonesia ke dalam dua kelas: **SPAM** atau **HAM** (Aman).

**Cobain Sekarang!** 
https://nolimit-farhanahmadn.streamlit.app/

## 📊 Dataset & License
Dataset yang digunakan adalah kumpulan teks *email* berbahasa Indonesia yang telah dilabeli sebagai *spam* atau *ham*. 
* **Sumber:** https://www.kaggle.com/datasets/gevabriel/indonesian-email-spam 
* **Lisensi:** *Open-source / Public Domain*. 
* **Sample Data:** *dataset* telah disertakan di dalam repositori ini (`data/indonesian_spam.csv`) untuk keperluan verifikasi lokal (*local verification*) tanpa perlu mengunduh data eksternal.

## 🧠 Architecture & Model Selection Justification
Sistem ini menggunakan ekosistem **Hugging Face** untuk ekstraksi *embedding* dan **Scikit-Learn** untuk klasifikasi. 

Daripada melakukan *fine-tuning* pada model Transformer berukuran masif (seperti IndoBERT standar), proyek ini menggunakan pendekatan *Feature Extraction Pipeline*:
1. **Embedding Layer:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
2. **Classification Layer:** `Scikit-Learn Logistic Regression`

**Kelebihan dan Justifikasi Pendekatan Ini:**
* **Efisiensi Komputasi & Inferensi Super Cepat:** Model `MiniLM` dirancang sangat ringan, memungkinkan proses *encoding* teks menjadi representasi vektor numerik (384-Dimensi) berjalan sangat cepat, bahkan tanpa akselerasi GPU.
* **Sentence-Level Representation:** Model *sentence-transformer* secara *out-of-the-box* menghasilkan vektor yang mewakili makna satu kalimat/paragraf utuh (melalui *pooling operation* internal), jauh lebih relevan untuk tugas klasifikasi teks panjang seperti *email* dibandingkan mengambil representasi token biasa.
* **Skalabilitas Deployment:** Dengan memisahkan proses *embedding* dan model regresi, ukuran file model akhir (`.pkl`) menjadi sangat kecil dan sangat optimal untuk di-*deploy* sebagai aplikasi web (*Streamlit*) tanpa risiko *Out-of-Memory* (OOM).

## 🗂️ Repository Structure
* `main.py` : Skrip utama berisi *pipeline* ekstraksi *embedding*, pelatihan model Sklearn, evaluasi (akurasi/F1-Score), dan pencetakan hasil.
* `app.py` : Skrip antarmuka web interaktif menggunakan Streamlit.
* `requirements.txt` : Daftar *library* dependensi. 
* `pipeline_flowchart.png` : Visualisasi *end-to-end pipeline* pemrosesan data.
* `spam_classifier_model.pkl` : Model klasifikasi Scikit-Learn yang telah dilatih.

## ⚙️ Setup & Installation Instructions 
Pastikan Anda memiliki Python versi 3.8 atau lebih baru.

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/username/nolimit-ds-test-farhan-ahmad.git](https://github.com/username/nolimit-ds-test-farhan-ahmad.git)
   cd nolimit-ds-test-farhan-ahmad
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan Training Pipeline**
   ```bash
   python main.py
   ```
4. **Jalankan Web**
   ```bash
   python -m streamlit run app.py
   ```

> *Developed by Farhan Ahmad Naufal for NoLimit Indonesia* 
