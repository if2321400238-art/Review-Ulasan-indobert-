import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# 1. SETTING HALAMAN UTAMA
st.set_page_config(page_title="Deteksi Anomali Ulasan - IndoBERT", layout="centered")
st.title("🛡️ Sistem Deteksi Anomali Ulasan (Powered by IndoBERT)")
st.write("Aplikasi ini mendeteksi apakah ulasan produk kosmetik termasuk Normal atau Anomali menggunakan Deep Learning.")

# 2. LOAD MODEL LANGSUNG DARI REPO HUGGING FACE + REPO GITHUB KAMU
@st.cache_resource
def load_my_clean_model():
    # File config.json & tokenizer dibaca dari folder lokal hasil push GitHub kamu
    local_folder = "indobert_final_model"
    tokenizer = BertTokenizer.from_pretrained(local_folder)
    
    # === ⚠️ GANTI INI DENGAN USERNAME HF & NAMA REPO MODEL HF KAMU ===
    # Contoh: "if2321400238/indobert-ulasan-kosmetik"
    hf_model_repo = "shahibkholil/indobert-ulasan-kosmetik"
    
    # Mengambil file config lokal tapi bobot modelnya otomatis disedot dari Hugging Face
    model = BertForSequenceClassification.from_pretrained(hf_model_repo, config=f"{local_folder}/config.json")
    return tokenizer, model

try:
    tokenizer, model = load_my_clean_model()
    st.success("✅ Model Cerdas IndoBERT Hasil Fine-Tuning Sukses Dimuat via Hugging Face Hub!")
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