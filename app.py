import streamlit as st
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Deteksi Ulasan Kosmetik Palsu",
    page_icon="🛡️",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🛡️ Deteksi Ulasan Kosmetik Palsu")
st.markdown(
    """
Aplikasi AI berbasis IndoBERT untuk mendeteksi:

- 🟢 Normal
- 🚨 Anomali
- 🤔 Skeptis

pada ulasan produk kosmetik.
"""
)

# =========================================================
# LABEL MAPPING
# =========================================================

label_map = {
    0: "NORMAL",
    1: "ANOMALI",
    2: "SKEPTIS"
}

label_emoji = {
    0: "🟢",
    1: "🚨",
    2: "🤔"
}

# =========================================================
# MODEL REPO
# =========================================================

MODEL_REPO = "shahibkholil/indobert-ulasan-kosmetik"

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO
    )

    model.eval()

    return tokenizer, model

# =========================================================
# LOAD
# =========================================================

try:

    tokenizer, model = load_model()

    st.success("✅ Model IndoBERT berhasil dimuat!")

except Exception as e:

    st.error(f"Gagal memuat model: {e}")

    st.stop()

# =========================================================
# EXAMPLE REVIEWS
# =========================================================

st.write("### 🧪 Contoh Ulasan")

examples = {
    "Normal": "Produk bagus, wanginya enak dan sesuai deskripsi.",
    "Anomali": "Parahhhh dapet yang palsu, wanginya beda banget.",
    "Skeptis": "Ini asli atau palsu sih? Kok teksturnya beda ya?"
}

selected_example = st.selectbox(
    "Pilih contoh ulasan:",
    [""] + list(examples.keys())
)

default_text = ""

if selected_example:
    default_text = examples[selected_example]

# =========================================================
# USER INPUT
# =========================================================

user_input = st.text_area(
    "Masukkan ulasan kosmetik:",
    value=default_text,
    placeholder="Contoh: wanginya beda banget dari official store...",
    height=150
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Analisis Ulasan"):

    if not user_input.strip():

        st.warning("Masukkan teks terlebih dahulu.")

    else:

        with st.spinner("Menganalisis ulasan..."):

            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():

                outputs = model(**inputs)

                logits = outputs.logits

                probs = torch.softmax(
                    logits,
                    dim=1
                ).cpu().numpy()[0]

                pred = int(np.argmax(probs))

        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("Hasil Analisis")

        st.markdown(
            f"""
## {label_emoji[pred]} {label_map[pred]}
"""
        )

        # =============================================
        # CONFIDENCE SCORE
        # =============================================

        st.write("### 📊 Confidence Score")

        st.write(f"🟢 NORMAL: {probs[0]*100:.2f}%")
        st.progress(float(probs[0]))

        st.write(f"🚨 ANOMALI: {probs[1]*100:.2f}%")
        st.progress(float(probs[1]))

        st.write(f"🤔 SKEPTIS: {probs[2]*100:.2f}%")
        st.progress(float(probs[2]))

        # =============================================
        # INTERPRETATION
        # =============================================

        st.write("---")

        if pred == 0:

            st.success(
                "Ulasan terdeteksi normal."
            )

        elif pred == 1:

            st.error(
                "Ulasan terindikasi anomali / kemungkinan produk palsu."
            )

        elif pred == 2:

            st.warning(
                "Ulasan bersifat skeptis atau mencurigakan."
            )

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.caption(
    "Powered by IndoBERT Fine-Tuning for Fake Cosmetic Review Detection"
)