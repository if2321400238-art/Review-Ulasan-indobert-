import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import requests

# 1. SETTING HALAMAN UTAMA
st.set_page_config(page_title="Deteksi Anomali Ulasan - IndoBERT", layout="centered")
st.title("🛡️ Sistem Deteksi Anomali Ulasan (Powered by IndoBERT)")
st.write("Aplikasi ini mendeteksi apakah ulasan produk kosmetik termasuk Normal atau Anomali menggunakan Deep Learning.")

# Fungsi pembantu untuk download file dari Google Drive secara otomatis
def download_from_drive(file_id, destination):
    if not os.path.exists(destination):
        with st.spinner(f"Mengunduh komponen model ({destination}) dari Google Drive server..."):
            url = f"https://docs.google.com/uc?export=download&id={file_id}"
            session = requests.Session()
            response = session.get(url, stream=True)
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)

# 2. PROSES DOWNLOAD & LOAD MODEL HASIL FINE-TUNING
@st.cache_resource
def load_my_fine_tuned_model():
    # Buat folder lokal di server Streamlit Cloud
    os.makedirs("my_model", exist_ok=True)
    
    # === ⚠️ GANTI ID DI BAWAH INI DENGAN ID UNIK DARI GOOGLE DRIVE KAMU ===
    CONFIG_ID = "ID_FILE_CONFIG_JSON_KAMU"
    VOCAB_ID = "ID_FILE_VOCAB_TXT_KAMU"
    MODEL_ID = "1LS_q-0YoydPmMVBkIk8pwYPNmopf1ZmN" # File seberat 498MB
    
    # Proses download otomatis via jalur belakang Google (Hanya sekali saat web pertama kali dinyalakan)
    download_from_drive(CONFIG_ID, "my_model/config.json")
    download_from_drive(VOCAB_ID, "my_model/vocab.txt")
    # Cek apakah modelnya safetensors atau pytorch_model.bin, sesuaikan namanya
    download_from_drive(MODEL_ID, "my_model/model.safetensors") 
    
    # Load model hasil download yang sudah pintar
    tokenizer = BertTokenizer.from_pretrained("my_model")
    model = BertForSequenceClassification.from_pretrained("my_model")
    return tokenizer, model

try:
    tokenizer, model = load_my_fine_tuned_model()
    st.success("✅ Model Cerdas IndoBERT Hasil Fine-Tuning Berhasil Dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model. Error: {e}")
    st.stop()

# 3. INPUT USER
user_input = st.text_area("Masukkan teks ulasan kosmetik di sini:", placeholder="Contoh: Ini beneran ori ga sih? Kok teksturnya beda banget... ")

# 4. PROSES PREDIKSI
if st.button("Analisis Ulasan dengan AI"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("IndoBERT sedang menganalisis konteks kalimat..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).flatten().numpy()
                prediction = np.argmax(probs)
            
            st.write("---")
            st.write(f"**Keyakinan Model (Normal):** {probs[0]*100:.2f}%")
            st.write(f"**Keyakinan Model (Anomali):** {probs[1]*100:.2f}%")
            
            if prediction == 1:
                st.error("🚨 **Hasil Analisis:** Ulasan Terdeteksi **ANOMALI / MENCURIGAKAN**")
            else:
                st.success("🟢 **Hasil Analisis:** Ulasan **NORMAL**")