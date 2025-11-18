import streamlit as st
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

st.set_page_config(page_title="AI Medical Prescription Verification", layout="wide")
st.title("AI Medical Prescription Verification")

# ---------------- IBM Watson Setup ----------------
watson_api_key = st.secrets.get("WATSON_API_KEY", "")
watson_url = st.secrets.get("WATSON_URL", "")

if watson_api_key and watson_url:
    authenticator = IAMAuthenticator(watson_api_key)
    nlu = NaturalLanguageUnderstandingV1(
        version='2023-11-17',
        authenticator=authenticator
    )
    nlu.set_service_url(watson_url)
else:
    st.warning("IBM Watson credentials not set in Streamlit secrets.")

# ---------------- Hugging Face Model Setup ----------------
@st.cache_resource
def load_hf_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # sentiment demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

hf_pipeline = load_hf_model()

# ---------------- Prescription Input ----------------
input_type = st.radio("Choose input type:", ["Text", "Image"])

if input_type == "Text":
    prescription_text = st.text_area("Enter prescription text here:")
    if prescription_text:
        # Hugging Face demo
        hf_result = hf_pipeline(prescription_text)
        st.write("Hugging Face Analysis:", hf_result)

        # IBM Watson NLU analysis
        if watson_api_key:
            try:
                watson_result = nlu.analyze(
                    text=prescription_text,
                    features=Features(entities=EntitiesOptions(), keywords=KeywordsOptions())
                ).get_result()
                st.write("IBM Watson NLU Analysis:", watson_result)
            except Exception as e:
                st.error(f"IBM Watson Analysis failed: {e}")

elif input_type == "Image":
    uploaded_file = st.file_uploader("Upload prescription image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Prescription", use_column_width=True)
        st.info("OCR and AI verification can be added here (Tesseract or Hugging Face Vision models)")

