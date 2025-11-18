import streamlit as st
from PIL import Image
import easyocr
from transformers import pipeline

st.set_page_config(page_title="AI Medical Prescription Verification", layout="wide")
st.title("AI Medical Prescription Verification (Hugging Face + OCR)")

# ---------------- Hugging Face Model ----------------
@st.cache_resource
def load_hf_model():
    # Using a simple classification pipeline for demonstration
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

hf_pipeline = load_hf_model()

# ---------------- OCR Setup ----------------
reader = easyocr.Reader(['en'])

# ---------------- Input ----------------
input_type = st.radio("Choose input type:", ["Text", "Image"])

if input_type == "Text":
    prescription_text = st.text_area("Enter prescription text here:")
    if prescription_text:
        st.subheader("Hugging Face Text Analysis")
        hf_result = hf_pipeline(prescription_text)
        st.write(hf_result)

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
            st.subheader("Hugging Face Text Analysis")
            hf_result = hf_pipeline(extracted_text)
            st.write(hf_result)
