import streamlit as st
from PIL import Image
import easyocr
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="AI Prescription Verification", layout="wide")
st.title("AI Prescription Verification (CPU-only)")

# Hugging Face pipeline
@st.cache_resource
def load_hf_model():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

hf_pipeline = load_hf_model()

# EasyOCR setup
reader = easyocr.Reader(['en'])

input_type = st.radio("Input type:", ["Text", "Image"])

if input_type == "Text":
    prescription_text = st.text_area("Enter prescription text:")
    if prescription_text:
        st.subheader("Text Analysis")
        st.write(hf_pipeline(prescription_text))

elif input_type == "Image":
    uploaded_file = st.file_uploader("Upload prescription image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Prescription", use_column_width=True)

        st.subheader("Extracted Text (OCR)")
        result = reader.readtext(np.array(image), detail=0)
        extracted_text = "\n".join(result)
        st.text(extracted_text)

        if extracted_text:
            st.subheader("Text Analysis")
            st.write(hf_pipeline(extracted_text))
