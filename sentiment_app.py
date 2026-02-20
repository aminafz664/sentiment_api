import streamlit as st
from transformers import pipeline

# لود مدل Hugging Face
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# عنوان سایت
st.title("Sentiment Analyzer 🌟")
st.write("این ابزار متن شما را تحلیل می‌کند و می‌گوید مثبت است یا منفی.")

# دریافت متن کاربر
user_input = st.text_area("متن خود را اینجا وارد کنید:")

# دکمه تحلیل
if st.button("Analyze"):
    if user_input.strip() != "":
        result = sentiment_model(user_input)[0]
        st.write(f"**Sentiment:** {result['label']}")
        st.write(f"**Confidence:** {round(result['score'],3)}")
    else:
        st.write("لطفاً متن خود را وارد کنید!")