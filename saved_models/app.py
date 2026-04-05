import streamlit as st
import pickle
import os

# Page Config
st.set_page_config(page_title="Spam Detector", page_icon="📧")

# Load the vectorizer
@st.cache_resource
def load_vectorizer():
    import os
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
    return pickle.load(open(vectorizer_path, 'rb'))

tfidf = load_vectorizer()

# UI Layout
st.title("📧 Multi-Model Spam Detector")
st.markdown("Enter an email or message below to check if it's **Spam** or **Ham**.")

# 1. Sidebar for Model Selection
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ("Naive_Bayes", "SVM", "Logistic_Regression", "Random_Forest")
)

# 2. Main Input Area
input_sms = st.text_area("Paste message here:", placeholder="Type something like 'You won a prize!'", height=150)

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Load chosen model
        import os
        model_path = os.path.join(os.path.dirname(__file__), f'{model_choice}.pkl')
        
        if os.path.exists(model_path):
            model = pickle.load(open(model_path, 'rb'))
            
            # Preprocess & Predict
            vectorized_input = tfidf.transform([input_sms])
            result = model.predict(vectorized_input)[0]
            
            # Display Result
            if result == 1:
                st.error(f"🚨 **SPAM DETECTED** (By {model_choice})")
            else:
                st.success(f"✅ **NOT SPAM / HAM** (By {model_choice})")
        else:
            st.error("Model file not found. Did you run train_models.py first?")

st.sidebar.info("This app compares 4 different algorithms to show how their logic affects classification.")