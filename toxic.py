import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
import time

# -----------------------------
# Sidebar: Project Info & Examples
# -----------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    **Toxic Comment Detection**
    
    Powered by DistilBERT. Enter a comment (or multiple, one per line) to analyze toxicity levels.
    
    - Model: DistilBERT fine-tuned for multi-label toxicity detection
    - Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
    - [GitHub](https://github.com/) | [Kaggle](https://kaggle.com/)
    """
)

st.sidebar.title("💡 Example Comments")
examples = [
    "You are so stupid!",
    "I hope you have a great day!",
    "I will find you and hurt you.",
    "This is an amazing project.",
    "You are a horrible person.",
    "I love your work!"
]
if st.sidebar.button("Insert Random Example"):
    import random
    st.session_state["comment_input"] = random.choice(examples)

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
    "Detect toxic comments using a **DistilBERT deep learning model**. Enter one or more comments (one per line) to analyze."
)



# -----------------------------
# File Uploader (user chooses a file by browsing)
# -----------------------------
st.markdown("---")
st.header("Analyze Comments from File")
file_comments = []
file_ids = []
df_file = None

uploaded_file = st.file_uploader(
    "Upload a CSV file with 'id' and 'comment_text' columns:",
    type=["csv"]
)
if uploaded_file is not None:
    df_file = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df_file)} comments from uploaded file.")


if df_file is not None:
    if 'comment_text' in df_file.columns:
        st.dataframe(df_file[['id', 'comment_text']], use_container_width=True)
        # Let user select comments to analyze
        st.markdown("**Select comments to analyze:**")
        selected_rows = st.multiselect(
            "Choose by index (row number):",
            options=list(df_file.index),
            default=list(df_file.index[:5]),
            format_func=lambda x: f"{x}: {df_file.loc[x, 'comment_text'][:50]}..."
        )
        if selected_rows:
            file_comments = df_file.loc[selected_rows, 'comment_text'].tolist()
            file_ids = df_file.loc[selected_rows, 'id'].tolist()
    else:
        st.error("CSV must contain a 'comment_text' column.")

# -----------------------------
# Text input (batch support)
# -----------------------------
st.markdown("---")
st.header("✍️ Or Enter Comments Manually")
text = st.text_area(
    "Enter comment(s) to analyze (one per line):",
    height=180,
    placeholder="Type or paste comments here...",
    key="comment_input"
)


# -----------------------------
# Prediction button (file or manual)
# -----------------------------
if st.button("Predict"):
    # Priority: file comments if present, else manual
    if file_comments:
        comments = file_comments
        ids = file_ids
    else:
        comments = [t.strip() for t in text.split("\n") if t.strip()]
        ids = None
    if not comments:
        st.warning("⚠️ Please enter or select at least one comment.")
    else:
        st.info(f"Analyzing {len(comments)} comment(s)...")
        all_probs = []
        verdicts = []
        for idx, comment in enumerate(comments):
            # Tokenize input
            inputs = tokenizer(
                comment,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64
            )
            # Model inference
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).squeeze().numpy()
            all_probs.append(probs)
            # Simple verdict: toxic if any label > 0.5
            is_toxic = np.any(probs > 0.5)
            verdicts.append("🛑 Toxic" if is_toxic else "✅ Not Toxic")
            time.sleep(0.05)  # For UI smoothness

        # Display results as DataFrame
        df = pd.DataFrame(all_probs, columns=LABELS)
        df["Verdict"] = verdicts
        df["Comment"] = comments
        if ids is not None:
            df["ID"] = ids
            display_cols = ["ID", "Comment"] + LABELS + ["Verdict"]
        else:
            display_cols = ["Comment"] + LABELS + ["Verdict"]
        st.subheader("📊 Toxicity Scores Table")
        st.dataframe(df[display_cols], use_container_width=True)


        # Visualization: Select any comment to view
        st.subheader("🔬 Visualization (Select Comment)")
        viz_idx = 0
        if len(comments) > 1:
            viz_idx = st.selectbox(
                "Select a comment to visualize:",
                options=list(range(len(comments))),
                format_func=lambda i: f"{comments[i][:50]}..."
            )
        chart_data = pd.DataFrame({"Label": LABELS, "Score": all_probs[viz_idx]})
        st.bar_chart(chart_data.set_index("Label"))

        # Show summary verdict for selected comment
        st.markdown(f"### Verdict: {verdicts[viz_idx]}")

        st.markdown("---")
        st.success("Prediction completed successfully ✅")

# -----------------------------
# Enhanced UI: More Attractive Touches
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f7fafd;
}
.stTextArea textarea {
    background-color: #f0f6ff;
    border-radius: 8px;
    font-size: 1.1em;
}
.stButton>button {
    color: white;
    background: linear-gradient(90deg, #1e90ff 0%, #00bfff 100%);
    border-radius: 8px;
    font-weight: bold;
    font-size: 1.1em;
}
.stDataFrame {
    background: #f0f6ff;
    border-radius: 8px;
}
.stAlert, .stSuccess, .stWarning {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center; font-size:1.3em; color:#1e90ff; margin-bottom:10px;'>🧠 Powered by DistilBERT & Transformers</div>",
    unsafe_allow_html=True
)

st.info("✨ Tip: You can paste multiple comments at once for batch analysis!")