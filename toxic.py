import streamlit as st
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# Debug: Confirm script is running
st.write("App loaded. If you see this, Streamlit is working.")

# -----------------------------
# Labels used in the project
# -----------------------------
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# -----------------------------
# Load model & tokenizer once
# -----------------------------
@st.cache_resource
def load_model():
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "toxicity_distilbert"
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            "toxicity_distilbert"
        )
        model.eval()
        st.write("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

tokenizer, model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Toxic Comment Detection")
st.write(
    "This app detects toxic comments using a **DistilBERT deep learning model**."
)

# -----------------------------
# Text input
# -----------------------------
text = st.text_area(
    "Enter a comment to analyze",
    height=120,
    placeholder="Type a comment here..."
)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter a comment.")
    else:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        )

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().numpy()

        # Display results
        st.subheader("📊 Toxicity Scores")

        for label, score in zip(LABELS, probs):
            st.write(f"**{label}** : {score:.2f}")

        st.markdown("---")
        st.success("Prediction completed successfully ✅")