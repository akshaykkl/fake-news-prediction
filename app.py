import streamlit as st
import joblib
import re
import string
import contractions
import emoji
import numpy as np
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')

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

# Initialize preprocessing tools
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"Failed to initialize NLP tools: {str(e)}")
    st.stop()

# Text cleaning function
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

# Load models with caching
@st.cache_resource
def load_traditional_model():
    try:
        model = joblib.load('model/baseline_model.joblib')
        vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
        if not hasattr(vectorizer, 'transform'):
            raise ValueError("Vectorizer is not properly initialized")
        return model, vectorizer
    except Exception as e:
        st.error(f"Traditional model loading error: {str(e)}")
        st.stop()

@st.cache_resource
def load_dnn_model():
    try:
        model = tf.keras.models.load_model('model/tf_text_classifier.keras')
        with open('model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"DNN model loading error: {str(e)}")
        st.stop()

# Prediction functions
def predict_traditional(text, model, vectorizer):
    try:
        cleaned_text = clean_text_advanced(text)
        if not cleaned_text.strip():
            raise ValueError("Text is empty after preprocessing")
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        return prediction, proba
    except Exception as e:
        raise ValueError(f"Traditional prediction failed: {str(e)}")

def predict_dnn(text, model, tokenizer, max_len=200):
    try:
        cleaned_text = clean_text_advanced(text)
        if not cleaned_text.strip():
            raise ValueError("Text is empty after preprocessing")
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding='post')
        proba = model.predict(padded, verbose=0)[0][0]
        prediction = 1 if proba >= 0.5 else 0
        return prediction, [1-proba, proba]  # [fake_prob, real_prob]
    except Exception as e:
        raise ValueError(f"DNN prediction failed: {str(e)}")

# Main app
def main():
    st.set_page_config(page_title="Real/Fake News Detector", page_icon="📰")
    st.title("📰 Real/Fake News Classifier")
    
    # Model selection
    model_type = st.radio("Select Model Type:",
                         ("Traditional (TF-IDF + Classifier)", "Deep Neural Network"),
                         index=0)
    
    # Load the selected model
    if model_type == "Traditional (TF-IDF + Classifier)":
        model, processor = load_traditional_model()
    else:
        model, processor = load_dnn_model()
    
    # User input
    user_input = st.text_area("Paste news article text here:", 
                            height=200,
                            placeholder="Enter the news content you want to analyze...")
    
    if st.button("Analyze Text"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing content..."):
                try:
                    if model_type == "Traditional (TF-IDF + Classifier)":
                        prediction, proba = predict_traditional(user_input, model, processor)
                    else:
                        prediction, proba = predict_dnn(user_input, model, processor)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Real Probability", value=f"{proba[1]:.1%}")
                        st.metric("Fake Probability", value=f"{proba[0]:.1%}")

                    with col2:
                        st.progress(float(proba[1]))  # Progress bar for real probability
                        st.caption(f"Real probability: {proba[1]:.1%}")
                        st.progress(float(proba[0]))  # Progress bar for fake probability
                        st.caption(f"Fake probability: {proba[0]:.1%}")

                    
                    # Show explanation
                    if prediction == 1:
                        st.success("""
                        **Authentic Content Indicators**
                        - Fact-based statements
                        - Verifiable sources
                        - Balanced perspective
                        """)
                    else:
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