import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Paths to models
PNEUMONIA_MODEL_FILENAME = r"lightweight_pneumonia_cnn_512.keras"
MAMMOGRAPHY_MODEL_FILENAME = r"lightweight_mammography_cnn_512.keras"
RNN_MCL_C_MODEL_FILENAME = r"lightweight_rnn_model_v2_1.keras"
RNN_TOKENIZER_FILENAME = r"tokenizer.pkl"          
RNN_LABEL_ENCODER = r"label_encoder.pkl"
# Load CNN models
try:
    pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_FILENAME)
    mammography_model = tf.keras.models.load_model(MAMMOGRAPHY_MODEL_FILENAME)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load RNN model, tokenizer and label encoder
try:
    rnn_model = tf.keras.models.load_model(RNN_MCL_C_MODEL_FILENAME)
    with open(RNN_TOKENIZER_FILENAME, 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open(RNN_LABEL_ENCODER, 'rb') as f:
        label_encoder = pickle.load(f)
        
    class_labels = ['Hyperthyroidism', 'PCOS', 'Hypothyroidism', 'Diabetes']
except Exception as e:
    st.error(f"Error loading RNN model or tokenizer: {e}")
    st.stop()

def predict_disease(text):
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=50, padding='post')
    probs = rnn_model.predict(pad_seq, verbose=0)[0]  

    class_probs = dict(zip(label_encoder.classes_, probs))
    
    most_likely_class = max(class_probs, key=class_probs.get)
    confidence = class_probs[most_likely_class]

    return most_likely_class, confidence

def predict_pneumonia(uploaded_file):
    img_size = (224, 224)
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_prob = pneumonia_model.predict(img_array)[0][0]
    predicted_class_index = (pred_prob > 0.5).astype(int)
    class_labels = ['Normal', 'Pneumonia']
    predicted_label = class_labels[predicted_class_index]
    confidence = pred_prob if predicted_class_index == 1 else (1 - pred_prob)
    return predicted_label, confidence

def predict_breast_cancer(uploaded_file):
    img_size = (224, 224)
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_prob = mammography_model.predict(img_array)[0][0]
    predicted_class_index = (pred_prob > 0.5).astype(int)
    class_labels = ['Benign', 'Malignant']
    predicted_label = class_labels[predicted_class_index]
    confidence = pred_prob if predicted_class_index == 1 else (1 - pred_prob)
    return predicted_label, confidence

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    main_choice = st.sidebar.selectbox("Select a category:", ["Medical Chatbot", "Radiologist"])

    if main_choice == "Medical Chatbot":
        st.header("Welcome to AIMAIC - AI for Medical Analysis and Image Classification")
        st.warning("""
        Disclaimer: This machine currently focuses on Hypo- and Hyperthyriodism, PCOS and Diabetes. More diseases will be added in the future.
        **Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition or the interpretation of your lab results.**
        If you are experiencing a medical emergency, call your local emergency number immediately.
        """)

        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Show chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("(I am angry all the time, I am cold all the time, My menstrual cycles are painful and irregular...)")
        if user_input:
            # Save user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Run prediction
            predicted_class, confidence = predict_disease(user_input)

            # Generate appropriate response
            if predicted_class is None:
                response = "I'm sorry, I couldn't identify the condition based on your symptoms. Could you please provide more details or consult a healthcare professional?"
            else:
                response = (
                    f"Prediction: **{predicted_class.capitalize()}** "
                    f"Confidence: {(confidence*100):.2f}%."
                )

            # Save and display bot response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    elif main_choice == "Radiologist":
        st.header("Radiologist: Image Analysis")
        model_type = st.sidebar.selectbox("Choose the analysis type:", ["Pneumonia X-ray Analysis", "Breast Cancer X-ray Analysis"])

        if model_type == "Pneumonia X-ray Analysis":
            st.subheader("Pneumonia X-ray Analysis")
            uploaded_file = st.file_uploader("Choose a X-ray image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                if st.button("Analyze X-ray"):
                    with st.spinner("Analyzing image... Please wait."):
                        try:
                            label, conf = predict_pneumonia(uploaded_file)
                            color = 'red' if label == 'Pneumonia' else 'green'
                            st.markdown(f"<h3 style='color: {color};'>Prediction: {label}</h3>", unsafe_allow_html=True)
                            st.write(f"Confidence: {conf*100:.2f}%")
                            st.write("Note: This is a machine learning-based prediction, not a definitive diagnosis.")
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")

        elif model_type == "Breast Cancer X-ray Analysis":
            st.subheader("Breast X-ray Analysis")
            uploaded_file = st.file_uploader("Choose a X-ray image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                if st.button("Analyze X-ray"):
                    with st.spinner("Analyzing image... Please wait."):
                        try:
                            label, conf = predict_breast_cancer(uploaded_file)
                            color = 'red' if label == 'Malignant' else 'green'
                            st.markdown(f"<h3 style='color: {color};'>Prediction: {label}</h3>", unsafe_allow_html=True)
                            st.write(f"Confidence: {conf*100:.2f}%")
                            st.write("Note: This is a machine learning-based prediction, not a definitive diagnosis.")
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")

# Run the app
if __name__ == "__main__":
    main()
