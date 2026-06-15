# [cite_start]NoLimit Indonesia Data Scientist Hiring Test [cite: 1]

[cite_start]**Repository Name:** `nolimit-ds-test-farhan-ahmad` 
[cite_start]**Task Option:** A (Classification - Spam/Ham Detection for Indonesian Emails) [cite: 5, 6]

[cite_start]Proyek ini dibangun untuk memenuhi kualifikasi tes Data Scientist di NoLimit Indonesia[cite: 1]. Sistem ini merupakan sebuah *end-to-end pipeline Machine Learning* untuk mengklasifikasikan teks *email* berbahasa Indonesia ke dalam dua kelas: **SPAM** atau **HAM** (Aman).

## 📊 Dataset & License
[cite_start]Dataset yang digunakan adalah kumpulan teks *email* berbahasa Indonesia yang telah dilabeli sebagai *spam* atau *ham*[cite: 13]. 
* [cite_start]**Sumber:** Data diperoleh dari sumber publik/Kaggle (kumpulan *email* bahasa Indonesia). [cite: 13]
* [cite_start]**Lisensi:** *Open-source / Public Domain*. [cite: 14]
* [cite_start]**Sample Data:** Sebagian kecil *dataset* telah disertakan di dalam repositori ini (`data/indonesian_spam.csv`) untuk keperluan verifikasi lokal (*local verification*) tanpa perlu mengunduh data eksternal[cite: 28].

## 🧠 Architecture & Model Selection Justification
[cite_start]Sistem ini mematuhi persyaratan wajib dengan menggunakan ekosistem **Hugging Face** untuk ekstraksi *embedding* [cite: 10] [cite_start]dan **Scikit-Learn** untuk klasifikasi[cite: 11]. 

Daripada melakukan *fine-tuning* pada model Transformer berukuran masif (seperti IndoBERT standar), proyek ini menggunakan pendekatan *Feature Extraction Pipeline*:
1. [cite_start]**Embedding Layer:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` [cite: 10, 12]
2. [cite_start]**Classification Layer:** `Scikit-Learn Logistic Regression` [cite: 11]

**Kelebihan dan Justifikasi Pendekatan Ini:**
* **Efisiensi Komputasi & Inferensi Super Cepat:** Model `MiniLM` dirancang sangat ringan, memungkinkan proses *encoding* teks menjadi representasi vektor numerik (384-Dimensi) berjalan sangat cepat, bahkan tanpa akselerasi GPU.
* **Sentence-Level Representation:** Model *sentence-transformer* secara *out-of-the-box* menghasilkan vektor yang mewakili makna satu kalimat/paragraf utuh (melalui *pooling operation* internal), jauh lebih relevan untuk tugas klasifikasi teks panjang seperti *email* dibandingkan mengambil representasi token biasa.
* **Skalabilitas Deployment:** Dengan memisahkan proses *embedding* dan model regresi, ukuran file model akhir (`.pkl`) menjadi sangat kecil dan sangat optimal untuk di-*deploy* sebagai aplikasi web (*Streamlit*) tanpa risiko *Out-of-Memory* (OOM).

## 🗂️ Repository Structure
* [cite_start]`main.py` : Skrip utama berisi *pipeline* ekstraksi *embedding*, pelatihan model Sklearn, evaluasi (akurasi/F1-Score), dan pencetakan hasil.
* [cite_start]`app.py` : Skrip antarmuka web interaktif menggunakan Streamlit (Poin Bonus).
* [cite_start]`requirements.txt` : Daftar *library* dependensi. [cite: 27]
* [cite_start]`pipeline_flowchart.png` : Visualisasi *end-to-end pipeline* pemrosesan data (Syarat Wajib)[cite: 15, 27].
* `spam_classifier_model.pkl` : Model klasifikasi Scikit-Learn yang telah dilatih.

## [cite_start]⚙️ Setup & Installation Instructions [cite: 12]
Pastikan Anda memiliki Python versi 3.8 atau lebih baru.

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/username/nolimit-ds-test-farhan-ahmad.git](https://github.com/username/nolimit-ds-test-farhan-ahmad.git)
   cd nolimit-ds-test-farhan-ahmad