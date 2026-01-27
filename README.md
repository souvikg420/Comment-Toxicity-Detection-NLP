# Comment-Toxicity-Detection-NLP
Deep learning–based multi-label toxic comment detection system using CNN, LSTM, and DistilBERT with Streamlit deployment for real-time and bulk predictions.
# 🛡️ Toxic Comment Detection using Deep Learning

This project implements an end-to-end **toxic comment detection system** using **Natural Language Processing (NLP)** and **Deep Learning**.  
Multiple models were trained and evaluated, and the best-performing model was deployed using **Streamlit**.

---

## 🚀 Project Overview

Online platforms often face challenges with abusive, hateful, and toxic comments.  
This project aims to **automatically detect toxic comments** to support content moderation systems.

The system classifies comments into **six toxicity categories**:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

This is a **multi-label classification problem**, as a single comment can belong to multiple categories.

---

## 🧠 Models Implemented

Three deep learning models were built and compared:

| Model | Description |
|-----|------------|
| CNN | Fast baseline model for phrase-level toxicity detection |
| LSTM | Sequence-based model to capture contextual dependencies |
| **DistilBERT** | Transformer-based model with contextual understanding (**final model**) |

---

## 📊 Model Evaluation

Models were evaluated using:
- Precision
- Recall
- F1-score
- **Macro F1-score** (primary metric)

### 🔥 Performance Summary

| Model | Macro F1-score |
|-----|---------------|
| CNN | ~0.55 |
| LSTM | ~0.35 |
| **DistilBERT** | **~0.66** ✅ |

**DistilBERT** was selected due to its superior performance, especially on rare toxicity classes such as `threat` and `identity_hate`.

---

## 🛠️ Tech Stack

- Python
- NLP (Tokenization, Text Cleaning)
- TensorFlow / Keras (CNN, LSTM)
- PyTorch & Hugging Face Transformers (DistilBERT)
- Scikit-learn (Evaluation)
- Streamlit (Web App Deployment)

---

## 📂 Project Structure

toxic-comment-detection-deep-learning/
│
├── data/
├── notebooks/
├── model/
├── app.py
├── requirements.txt
├── README.md



---

## 🌐 Streamlit Web Application



The Streamlit app allows:
- Real-time toxicity prediction for a single comment
- Bulk prediction via CSV upload
- Display of toxicity scores for each category

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py

Dataset

Dataset used:

Jigsaw Toxic Comment Classification Dataset

Each comment is annotated with six toxicity labels.


🎯 Key Learnings


Multi-label text classification

Deep learning model comparison

Handling class imbalance in NLP

Transformer-based model fine-tuning

Deploying NLP models using Streamlit


📌 Future Improvements


Threshold tuning per label

Model explainability (SHAP / LIME)

Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

Active learning for continuous improvement


👤 Author


Souvik Ghosh
Aspiring Data Scientist / NLP Engineer

⭐ Acknowledgements


Kaggle Jigsaw Toxic Comment Dataset

Hugging Face Transformers

Google Colab
