"""
app.py
======
Lightweight web GUI for the deployed surface-texture classifier (brief
Section 3.8). Built with Streamlit so a marker can drag-and-drop a texture
image and instantly see the predicted material class and confidence.

Run:
    pip install streamlit
    streamlit run src/app.py

If you prefer no web dependency, predict.py provides an equivalent CLI.
"""
from pathlib import Path
import numpy as np

import streamlit as st          # noqa: E402

import config
import utils
import preprocess as pp
import features as ft
from predict import load_model

st.set_page_config(page_title="Texture Classifier", page_icon="🧵", layout="centered")
st.title("Surface Material Texture Classifier")
st.caption("CT104-3-M Pattern Recognition — classical feature-engineering pipeline (KTH-TIPS2-b)")


@st.cache_resource
def _get_model(name):
    return load_model(None if name == "best" else name)


model_choice = st.selectbox("Model", ["best", "svm", "random_forest", "knn", "mlp"])
name, model, feature_set = _get_model(model_choice)
st.write(f"**Active model:** `{name}`  |  **feature set:** `{feature_set}`")

uploaded = st.file_uploader("Upload a surface-texture image",
                            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is not None:
    import cv2
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input", use_column_width=True)

    proc = pp.preprocess(img)
    col2.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB), caption="Pre-processed",
               use_column_width=True)

    x = ft.extract(proc, feature_set).reshape(1, -1)
    pred = int(model.predict(x)[0])
    st.subheader(f"Prediction: {config.CLASSES[pred]}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        st.bar_chart({config.CLASSES[i]: float(proba[i]) for i in range(len(config.CLASSES))})
        st.write(f"Confidence: **{proba[pred]*100:.1f}%**")
