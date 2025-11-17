import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests


tokenizer = AutoTokenizer.from_pretrained("Muizzzz8/phi3-prescription-reader")
model = AutoModelForCausalLM.from_pretrained("Muizzzz8/phi3-prescription-reader")

st.title("AI Prescription Verification App")

uploaded_file = st.file_uploader("Upload Prescription Image or enter text below", type=["jpg", "jpeg", "png", "txt"])

extracted_text = ""
if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Prescription Image", use_column_width=True)
       
        extracted_text = st.text_area("OCR Result", value="Paracetamol 500mg, Take 1 tablet every 6 hours")
    elif uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        st.write("Uploaded Prescription Text:")
        st.write(content)
        extracted_text = content

user_input = st.text_area("Or manually input prescription text here", value=extracted_text)

if st.button("Extract & Validate Prescription"):
    with st.spinner("Extracting information..."):
        prompt = f"Read the following prescription and extract medicine names and dosages:\n{user_input}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=200)
        extracted_entities = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Extracted Medicines/Dosages")
        st.write(extracted_entities)

      
        ibm_api_key = "YOUR_WATSON_API_KEY"
        ibm_url = "https://your_watson_instance/api/validate_prescription"
        payload = {
            "prescription_data": extracted_entities,
            "patient_info": {"age": 50, "weight": 80, "known_allergies": []}  
        }
        headers = {
            "Authorization": f"Bearer {ibm_api_key}",
            "Content-Type": "application/json"
        }
       
        st.info("Replace the IBM Watson API call with your real credentials and handle the API output.")

