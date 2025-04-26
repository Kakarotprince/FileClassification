# app.py
import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
from gensim.models import KeyedVectors, Doc2Vec
from sklearn.preprocessing import LabelBinarizer
from utils import extract_text_from_file, clean_text, w2v_tokenize_text, word_averaging_list, infer_vector, WeightedVotingEnsemble

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Weighted Voting Ensemble Class ---
from sklearn.base import BaseEstimator, ClassifierMixin



# Load models
@st.cache_resource
def load_models():
    svm = pickle.load(open('models/svm_model.pkl', 'rb'))
    ada = pickle.load(open('models/ada_model.pkl', 'rb'))
    rf = pickle.load(open('models/rfmodel.pkl', 'rb'))
    xgb = pickle.load(open('models/xgbmodel.pkl', 'rb'))
    nb = pickle.load(open('models/nbmodel.pkl', 'rb'))
    w2v = pickle.load(open('models/w2vmodel.pkl', 'rb'))
    d2v = pickle.load(open('models/d2vmodel.pkl', 'rb'))
    tfidf = pickle.load(open("models/tfidfconverter.pkl", "rb"))
    labelenc = pickle.load(open("models/labelencoder.pkl", 'rb'))
    doc2vec_model = Doc2Vec.load("models/doc2vec_dbow.model")
    word2vec_model = KeyedVectors.load("GoogleNews-vectors.kv")
    return svm, ada, rf, xgb, nb, w2v, d2v, tfidf, labelenc, doc2vec_model, word2vec_model

# Load all cached models
(svm_model, ada_model, rfmodel, xgbmodel, nbmodel,
 w2vmodel, d2vmodel, tfidfconverter, labelencoder,
 model_dbow, wv) = load_models()


# Setup ensemble model
models = [svm_model, ada_model, rfmodel, nbmodel, xgbmodel, w2vmodel, d2vmodel]
weights = [0.9139, 0.9055, 0.9292, 0.8474, 0.9373, 0.8237, 0.8322]

lb = LabelBinarizer()
categories = labelencoder.classes_
lb.fit(labelencoder.transform(categories))

ensemble_model = WeightedVotingEnsemble(models, weights, lb, labels=categories)

# Streamlit UI
st.title("\U0001F9E0 ML File Classifier")
uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'xlsx', 'txt', 'html', 'jpg', 'png','pptx','ppt'])

if uploaded_file is not None:
    file_path = uploaded_file.name
    st.info("Extracting text...")
    text = extract_text_from_file(file_path)
    st.text_area("Extracted Text", text, height=200)

    # TFIDF input
    tdf = pd.DataFrame([text], columns=['data'])
    tdf["text"] = tdf['data'].astype(str).apply(clean_text)
    inputs = pd.DataFrame(tfidfconverter.transform(tdf['data']).toarray())

    st.markdown("### \U0001F50D Predictions")
    st.write(f"**Support Vector Machine**: {labelencoder.inverse_transform(svm_model.predict(inputs))[0]}")
    st.write(f"**ADABOOST**: {labelencoder.inverse_transform(ada_model.predict(inputs))[0]}")
    st.write(f"**Random Forest**: {labelencoder.inverse_transform(rfmodel.predict(inputs))[0]}")
    st.write(f"**XGBoost**: {labelencoder.inverse_transform(xgbmodel.predict(inputs))[0]}")
    st.write(f"**Naive Bayes**: {labelencoder.inverse_transform(nbmodel.predict(inputs))[0]}")

    # Word2Vec input
    cleaned_text = clean_text(text)
    w2v_tokens = w2v_tokenize_text(cleaned_text)
    w2v_input = word_averaging_list(wv, [w2v_tokens])
    st.write(f"**Word2Vec + Logistic Regression**: {w2vmodel.predict(w2v_input)[0]}")

    # Doc2Vec input
    d2v_input = infer_vector(cleaned_text, model_dbow).reshape(1, -1)
    st.write(f"**Doc2Vec + Logistic Regression**: {d2vmodel.predict(d2v_input)[0]}")

    # Ensemble prediction
    X_list = [
        inputs,      # for SVM
        inputs,      # for AdaBoost
        inputs,      # for Random Forest
        inputs,      # for Naive Bayes
        inputs,      # for XGBoost
        w2v_input,   # for Word2Vec
        d2v_input    # for Doc2Vec
    ]

    ensemble_pred = ensemble_model.predict(X_list)[0]
    ensemble_category = labelencoder.inverse_transform([ensemble_pred])[0]

    # Soft Voting Ensemble (now includes SVM)
    soft_models = [svm_model, ada_model, rfmodel, nbmodel, xgbmodel, w2vmodel, d2vmodel]
    soft_weights = [0.9139, 0.9055, 0.9292, 0.8474, 0.9373, 0.8237, 0.8322]
    soft_inputs = [inputs, inputs, inputs, inputs, inputs, w2v_input, d2v_input]

    soft_ensemble = WeightedVotingEnsemble(soft_models, soft_weights, lb, labels=categories, voting='soft')
    soft_pred = soft_ensemble.predict(soft_inputs)
    soft_category = labelencoder.inverse_transform(soft_pred)[0]

    st.markdown("### 🧠 Ensemble Model Prediction")
    st.write(f"**Weighted Voting Ensemble**: {ensemble_category}")
    st.markdown("### 🤖 Soft Voting Ensemble Prediction")
    st.write(f"**Soft Voting Ensemble**: {soft_category}")