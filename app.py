import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Document Classifier", page_icon="📄", layout="centered")
st.title("📄 Automatic Document Image Classifier")

@st.cache_resource
def load_model():
    # Loads your custom model from the local folder
    return pipeline(
           "image-classification", 
           model="./my_document_classifier",
           image_processor="microsoft/dit-base"
       )

with st.spinner("Loading custom model..."):
    classifier = load_model()

uploaded_file = st.file_uploader("Upload a document image (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Document", width="stretch")

    with col2:
        st.subheader("Classification Results")
        with st.spinner("Analyzing..."):
            predictions = classifier(image)
            st.success(f"**Predicted Category:** {predictions[0]['label'].upper()}")
            st.metric(label="Confidence", value=f"{predictions[0]['score'] * 100:.2f}%")

            st.write("**Alternative Predictions:**")
            for pred in predictions[1:3]:
                st.write(f"- {pred['label'].capitalize()}: {pred['score'] * 100:.2f}%")