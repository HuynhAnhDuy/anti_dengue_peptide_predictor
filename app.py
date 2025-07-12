import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import sys
from tensorflow.keras.models import load_model
from PIL import Image

from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, Flatten,
    Bidirectional, LSTM, Dense, Dropout,
    Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)

# ====== Google Analytics Tracking (GA4) ======
GA_JS = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-Y27P57QD3C"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-Y27P57QD3C');
</script>
"""
st.markdown(GA_JS, unsafe_allow_html=True)

# ====== 1. Load Preprocessing Config ======
@st.cache_resource
def load_preprocessing(path="preprocessing.pkl"):
    with open(path, "rb") as f:
        config = pickle.load(f)
    return config["kmer_to_index"], config["max_length"]

# ====== 2. Load Trained Models ======
@st.cache_resource
def load_models():
    missing = [name for name in ['cnn','bilstm','transformer','meta'] if not os.path.exists(f"{name}.keras")]
    if missing:
        st.error(f"Missing models: {', '.join(missing)}. Please make sure all model files are available.")
        st.stop()
    return {name: load_model(f"{name}.keras") for name in ['cnn','bilstm','transformer','meta']}

# ====== 3. Predict from Sequence ======
def predict_sequence(seq, kmer_to_index, max_len, models):
    k = len(next(iter(kmer_to_index)))
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    x = np.zeros((1, max_len), dtype=np.int32)
    for j, kmer in enumerate(kmers[:max_len]):
        x[0, j] = kmer_to_index.get(kmer, 0)
    preds = [model.predict(x).item() for model in [models['cnn'], models['bilstm'], models['transformer']]]
    meta_input = np.array(preds).reshape(1, -1)
    final = models['meta'].predict(meta_input).item()
    return preds, final

# ====== 4. Predict Batch from DataFrame ======
def predict_batch(df, kmer_to_index, max_len, models):
    results = []
    for seq in df["Sequence"]:
        preds, final = predict_sequence(seq, kmer_to_index, max_len, models)
        results.append({
            "Sequence": seq,
            "CNN": preds[0],
            "BiLSTM": preds[1],
            "Transformer": preds[2],
            "MetaScore": final,
            "Prediction": "Anti-Dengue Peptide" if final >= 0.5 else "Non-Anti-Dengue Peptide"
        })
    return pd.DataFrame(results)

# ====== 5. Streamlit UI ======
st.title("🦠 Anti-Dengue Peptide Predictor")
st.markdown("### 📝 Description")
st.markdown("This tool predicts whether a given peptide sequence has potential anti-dengue activity using a stacking ensemble of neural networks: Convolutional Neural Network (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and Transformer.")
st.markdown("Sequence representations: K-mer fingerprints (K = 3)")
# === Sidebar instructions ===
with st.sidebar:
    st.header("📘 Instructions")
    st.markdown("""
    **Single prediction:**  
    - Enter 1 peptide sequence and click **Predict**

    **Batch prediction:**  
    - Upload a CSV file with a column named `Sequence`  
    - View and download prediction results
    """)
    st.markdown("### 📊 Classification Rule")
    st.markdown("- Meta-model score ≥ 0.5 → 🧬 *Anti-Dengue Peptide*  \n- < 0.5 → 🚫 *Non-Anti-Dengue Peptide*")

# === Single prediction ===
st.subheader("🦠 Single sequence prediction")
seq = st.text_area("Enter a peptide sequence:", height=100)
if st.button("Predict") and seq:
    k2i, ml = load_preprocessing("preprocessing.pkl")
    models = load_models()
    base_scores, score = predict_sequence(seq.strip(), k2i, ml, models)
    st.write("**Base model probabilities:**")
    st.write(f"CNN: {base_scores[0]:.4f}, BiLSTM: {base_scores[1]:.4f}, Transformer: {base_scores[2]:.4f}")
    st.write(f"**Meta-model final score:** {score:.4f}")
    st.write("**Prediction:**", "🧬 Anti-Dengue Peptide" if score > 0.5 else "🚫 Non-Anti-Dengue Peptide")

# === Batch prediction ===
st.subheader("🦠 Batch prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with a 'Sequence' column", type=["csv"])
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        if "Sequence" not in df_input.columns:
            st.error("CSV must contain a 'Sequence' column.")
        else:
            k2i, ml = load_preprocessing("preprocessing.pkl")
            models = load_models()
            df_result = predict_batch(df_input, k2i, ml, models)
            st.write("### Prediction Results")
            st.dataframe(df_result)

            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv, file_name="ADP_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# === About the Authors ===
st.markdown("---")
st.subheader("👨‍🔬 About the Authors")

col1, col2 = st.columns(2)

with col1:
    image1 = Image.open("assets/duy.jpg")
    st.image(image1, caption="Huynh Anh Duy", width=160)
    st.markdown("""
    **Huynh Anh Duy**  
    Can Tho University, Vietnam  
    PhD Candidate, Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [huynhanhduy.h@kkumail.com](mailto:huynhanhduy.h@kkumail.com), [haduy@ctu.edu.vn](mailto:haduy@ctu.edu.vn)
    """)

with col2:
    image2 = Image.open("assets/tarasi.png")
    st.image(image2, caption="Tarapong Srisongkram", width=160)
    st.markdown("""
    **Asst Prof. Dr. Tarapong Srisongkram**  
    Faculty of Pharmaceutical Sciences  
    Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [tarasri@kku.ac.th](mailto:tarasri@kku.ac.th)
    """)

# === Footer ===
st.markdown("---")
st.caption(f"🔧 Python version: {sys.version.split()[0]}")

# ====== 6. Run ======
# streamlit run app.py