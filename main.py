import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load Dataset (Sesuaikan nama file dan nama kolom dengan dataset Anda)
# Asumsi: dataset memiliki kolom 'text' (isi email) dan 'label' (spam/ham)
print("Memuat dataset...")
df = pd.read_csv("data/email_spam_indo.csv") 

X_text = df['Pesan'].tolist()
y_labels = df['Kategori'].tolist()

# 2. Extract Embeddings (Syarat Wajib menggunakan Hugging Face)
# Menggunakan model multilingual ringan agar proses lokal cepat
print("Memuat model Hugging Face dan mengekstrak embeddings...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_embeddings = model.encode(X_text, show_progress_bar=False)

# 3. Split Data
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X_embeddings, y_labels, X_text, test_size=0.2, random_state=42
)

# 4. Melatih Model Klasifikasi (Syarat Wajib menggunakan Sklearn/ANN)
print("Melatih model klasifikasi...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 5. Prediksi dan Evaluasi Minimum Output
print("\n--- Hasil Evaluasi ---")
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Menampilkan Contoh Output (Syarat Minimum Output)
print("\n--- Contoh Prediksi ---")
for i in range(5):
    print(f"Teks Asli: {text_test[i][:100]}...") # Tampilkan 100 karakter pertama
    print(f"Prediksi: {y_pred[i]} | Label Asli: {y_test[i]}")
    print("-" * 30)

# Simpan model yang sudah dilatih
joblib.dump(classifier, "spam_classifier_model.pkl")
print("Model berhasil disimpan sebagai spam_classifier_model.pkl!")