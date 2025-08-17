import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load resources
@st.cache_resource  # Cache to avoid reloading on every interaction
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_assets()

# App UI
st.title("✉️ Email Spam Detector")
user_input = st.text_area("Paste email text here:", height=150)

if st.button("Check Spam"):
    if user_input:
        try:
            # Transform input
            X = vectorizer.transform([user_input])  # Note: List wrapper
            
            # Predict
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]  # Confidence scores
            
            # Display results
            if prediction == 1:
                st.error(f"🚨 SPAM ({(proba[1]*100):.1f}% confidence)")
            else:
                st.success(f"✅ HAM ({(proba[0]*100):.1f}% confidence)")
                
            st.write("")  # Spacer
            st.progress(max(proba))  # Visual confidence indicator
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some text")