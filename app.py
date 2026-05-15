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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.result-box {
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
}

.small-text {
    font-size: 14px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🛡️ Deteksi Ulasan Kosmetik Palsu")

st.markdown("""
Sistem AI berbasis **IndoBERT** untuk mendeteksi:

- 🟢 **Normal**
- 🤔 **Skeptis**
- 🚨 **Anomali**

pada ulasan produk kosmetik.
""")

# =========================================================
# LABEL MAPPING
# =========================================================

label_map = {
    0: "NORMAL",
    1: "SKEPTIS",
    2: "ANOMALI"
}

label_emoji = {
    0: "🟢",
    1: "🤔",
    2: "🚨"
}

label_color = {
    0: "success",
    1: "warning",
    2: "error"
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
# LOAD MODEL
# =========================================================

try:

    tokenizer, model = load_model()

    st.success("✅ Model IndoBERT berhasil dimuat")

except Exception as e:

    st.error(f"Gagal memuat model: {e}")

    st.stop()

# =========================================================
# EXAMPLES
# =========================================================

st.write("### 🧪 Contoh Ulasan")

examples = {
    "🟢 Normal":
        "Produk bagus, wanginya enak dan sesuai deskripsi.",

    "🤔 Skeptis":
        "Ini asli atau palsu sih? Kok teksturnya beda ya?",

    "🚨 Anomali":
        "Barangnya palsu, wanginya beda dan packaging tidak sesuai."
}

selected_example = st.selectbox(
    "Pilih contoh:",
    [""] + list(examples.keys())
)

default_text = ""

if selected_example:
    default_text = examples[selected_example]

# =========================================================
# INPUT
# =========================================================

user_input = st.text_area(
    "Masukkan ulasan kosmetik:",
    value=default_text,
    placeholder="Contoh: wanginya beda banget dari official store...",
    height=170
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Analisis Ulasan", use_container_width=True):

    if not user_input.strip():

        st.warning("Masukkan teks terlebih dahulu.")

    else:

        with st.spinner("Model IndoBERT sedang menganalisis..."):

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

                confidence = probs[pred] * 100

        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("📌 Hasil Analisis")

        result_text = f"{label_emoji[pred]} {label_map[pred]}"

        if pred == 0:

            st.success(
                f"{result_text} ({confidence:.2f}%)"
            )

        elif pred == 1:

            st.warning(
                f"{result_text} ({confidence:.2f}%)"
            )

        elif pred == 2:

            st.error(
                f"{result_text} ({confidence:.2f}%)"
            )

        # =================================================
        # INTERPRETATION
        # =================================================

        st.write("### 🧠 Interpretasi")

        if pred == 0:

            st.write(
                """
Ulasan terdeteksi **normal** dan tidak menunjukkan
indikasi kuat terhadap produk palsu.
"""
            )

        elif pred == 1:

            st.write(
                """
Ulasan mengandung unsur **keraguan atau kecurigaan**
terhadap keaslian produk.
"""
            )

        elif pred == 2:

            st.write(
                """
Ulasan terindikasi kuat mengarah pada
**produk palsu / tidak original**.
"""
            )

        # =================================================
        # CONFIDENCE SCORE
        # =================================================

        st.write("### 📊 Confidence Score")

        st.write(f"🟢 NORMAL : {probs[0]*100:.2f}%")
        st.progress(float(probs[0]))

        st.write(f"🤔 SKEPTIS : {probs[1]*100:.2f}%")
        st.progress(float(probs[1]))

        st.write(f"🚨 ANOMALI : {probs[2]*100:.2f}%")
        st.progress(float(probs[2]))

        # =================================================
        # TOP PREDICTION
        # =================================================

        st.write("### 🎯 Prediksi Utama")

        st.info(
            f"""
Model paling yakin bahwa ulasan ini termasuk kategori:
**{label_map[pred]}**
dengan confidence **{confidence:.2f}%**
"""
        )

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.caption(
    "Powered by IndoBERT Fine-Tuning • Fake Cosmetic Review Detection"
)