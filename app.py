import streamlit as st
import joblib
import re
import string
import contractions
import emoji
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize NLTK with error handling
def initialize_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('stopwords')
            nltk.download('wordnet')
        except Exception as e:
            st.error(f"Failed to initialize NLTK: {str(e)}")
            st.stop()

initialize_nltk()

# Initialize preprocessing tools with error handling
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"Failed to initialize NLP tools: {str(e)}")
    st.stop()

def clean_text_advanced(text):
    try:
        text = str(text).lower()
        text = contractions.fix(text)
        text = emoji.replace_emoji(text, '')
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\w*\d\w*', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"Text cleaning failed: {str(e)}")
        return ""

@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/baseline_model.joblib')
        vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
        
        # Verify vectorizer is properly loaded
        if not hasattr(vectorizer, 'transform'):
            raise ValueError("Vectorizer is not properly initialized")
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.stop()

def predict(text, model, vectorizer):
    try:
        cleaned_text = clean_text_advanced(text)
        if not cleaned_text.strip():
            raise ValueError("Text is empty after preprocessing")
            
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        return prediction, proba
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

def main():
    st.set_page_config(page_title="Real/Fake News Detector", page_icon="📰")
    st.title("📰 Real/Fake News Classifier")
    
    # Load models
    model, vectorizer = load_models()
    
    # User input
    user_input = st.text_area("Paste news article text here:", height=200, 
                            placeholder="Enter the news content you want to analyze...")
    
    if st.button("Analyze Text"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing content..."):
                try:
                    prediction, proba = predict(user_input, model, vectorizer)
                    
                    # Display results - NOTE: Now 1=Real, 0=Fake
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Changed to match your labeling scheme
                        label = "REAL ✅" if prediction == 1 else "FAKE ❌"
                        confidence = proba[1] if prediction == 1 else proba[0]
                        st.metric("Prediction", value=label, delta=f"{confidence:.1%} confidence")
                    
                    with col2:
                        # Progress bar shows confidence in the predicted class
                        st.progress(confidence)
                        st.caption(f"Real probability: {proba[1]:.1%} | Fake probability: {proba[0]:.1%}")
                    
                    # Show explanation - updated to match your labels
                    if prediction == 1:  # Real
                        st.success("""
                        **Authentic Content Indicators**
                        - Fact-based statements
                        - Verifiable sources
                        - Balanced perspective
                        """)
                    else:  # Fake
                        st.warning("""
                        **Warning: Potential Fake Content Detected**
                        - Emotional or exaggerated language
                        - Lack of credible sources
                        - Inconsistent claims
                        """)
                        st.info("ℹ️ Verify with fact-checking websites before sharing")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Try entering a longer text with more meaningful content")

if __name__ == "__main__":
    main()