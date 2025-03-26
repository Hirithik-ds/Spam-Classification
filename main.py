import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the trained model
model = joblib.load("spam_classifier.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Sidebar settings
st.sidebar.title("🔧 Settings")
show_wordcloud = st.sidebar.checkbox("Show Spam Word Cloud", value=True)

# Main title
st.title("📩 Spam Detection App")
st.write("Enter a message to check if it's Spam or Not Spam.")

# User text input
user_input = st.text_area("✍️ Enter your message:", "")

if st.button("🔍 Predict"):
    if user_input:
        cleaned_input = clean_text(user_input)
        prob = model.predict_proba([cleaned_input])[:,1]  # Get spam probability
        threshold = 0.3  # Lower the threshold to detect more spam
        prediction = 1 if prob > threshold else 0  # Classify based on new threshold
        confidence = prob[0]  # Confidence score

        if prediction == 1:
            st.error(f"🚨 **Spam!** (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ **Not Spam!** (Confidence: {confidence:.2f})")
    else:
        st.warning("⚠️ Please enter a message to classify.")

# Word Cloud for Spam Messages
if show_wordcloud:
    spam_words = "win money lottery free offer congratulations claim"
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(spam_words)

    st.subheader("📊 Common Spam Words")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Batch Prediction: Upload CSV
st.subheader("📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'message' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "message" in df.columns:
        df["cleaned_text"] = df["message"].apply(clean_text)
        df["prediction"] = model.predict(df["cleaned_text"])
        df["prediction_label"] = df["prediction"].map({0: "Not Spam", 1: "Spam"})

        st.dataframe(df[["message", "prediction_label"]])
        st.download_button("⬇️ Download Predictions", df.to_csv(index=False), "spam_predictions.csv", "text/csv")
    else:
        st.error("❌ CSV must contain a 'message' column.")