import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title(" Fake News Detection App")
st.caption("Check whether a news article is real or fake using Machine Learning")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# OPTION 1 — Manual input
st.subheader(" Enter News Manually")
text = st.text_area("Paste your news here:", height=150)

# OPTION 2 — File upload
st.subheader(" Or Upload a .txt File")
uploaded_file = st.file_uploader("Upload file", type=["txt"])

file_text = ""

if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8")
    st.success("File uploaded successfully ")
    st.write(file_text[:500])  # preview first 500 chars

# Button
if st.button(" Analyze News"):

    final_text = ""

    if text.strip() != "":
        final_text = text
    elif file_text != "":
        final_text = file_text
    else:
        st.warning(" Please enter text or upload a file")
    
    if final_text != "":
        vector = vectorizer.transform([final_text])
        prediction = model.predict(vector)
        prob = model.predict_proba(vector)

        confidence = max(prob[0]) * 100

        if prediction[0] == 1:
            st.success(f" REAL News\nConfidence: {confidence:.2f}%")
        else:
            st.error(f" FAKE News\nConfidence: {confidence:.2f}%")

# Sidebar
st.sidebar.title(" About")
st.sidebar.write("""
- Detects Fake News using ML  
- Supports manual input + file upload  
""")

# Footer
st.markdown("---")
st.write("###  How it works")
st.write("""
1. Input text or upload file  
2. Convert to TF-IDF  
3. Predict using ML model  
""")