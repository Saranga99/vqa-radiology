import streamlit as st
from PIL import Image
from src.inference import ask_question

st.title("🩻 Medical VQA (VQA-RAD) Demo")

uploaded_file = st.file_uploader("Upload a Medical Image", type=["jpg", "png"])
question = st.text_input("Enter your question:")

if uploaded_file and question:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    answer = ask_question(uploaded_file, question)
    st.subheader(f"Answer: {answer}")