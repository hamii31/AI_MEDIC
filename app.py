import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from transformers import pipeline

# Paths and filenames
PNEUMONIA_MODEL_FILENAME = r"lightweight_pneumonia_cnn_512.keras"

# Load the pneumonia detection model
try:
    pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_FILENAME)
except FileNotFoundError as e:
    st.error(f"Error loading pneumonia model: {e}. Make sure the model exists at {PNEUMONIA_MODEL_FILENAME}.")
    st.stop()
except Exception as e:
    st.error(f"Error loading pneumonia model: {e}")
    st.stop()

# Load Hugging Face NLP pipeline for sentiment analysis
try:
    hf_classifier = pipeline("sentiment-analysis")
except Exception as e:
    st.error(f"Error loading Hugging Face NLP pipeline: {e}")
    st.stop()

# --- Medical Knowledge Base ---
symptoms_list = [
    "fever", "cough", "sore throat", "headache", "fatigue",
    "shortness of breath", "chest pain", "nausea", "vomiting",
    "diarrhea", "abdominal pain", "rash", "dizziness", "loss of taste", "loss of smell",
    "increased thirst", "frequent urination", "unexplained weight loss", "blurred vision",
    "weight gain", "weight loss", "hair loss", "dry skin", "feeling cold", "feeling hot", "palpitations",
    "irregular periods", "acne", "excess hair growth", "weight gain (especially around the middle)"
]

symptom_condition_mapping = {
    "fever": ["flu", "common cold", "infection"],
    "cough": ["flu", "common cold", "bronchitis"],
    "sore throat": ["common cold", "strep throat", "sore throat"],
    "headache": ["tension headache", "migraine", "flu"],
    "shortness of breath": ["asthma", "pneumonia", "anxiety"],
    "chest pain": ["heart attack (seek immediate medical help!)", "anxiety", "muscle strain"],
    "increased thirst": ["diabetes"],
    "frequest urination": ["diabetes"],
    "unexplained weight loss": ["diabetes", "hyperthyroidism"],
    "blurred vision": ["diabetes"],
    "weight gain": ["hypothyroidism", "PCOS"],
    "weight loss": ["hyperthyroidism"],
    "hair loss": ["hypothyroidism", "PCOS"],
    "dry skin": ["hypothyroidism"],
    "feeling cold": ["hypothyroidism"],
    "feeling hot": ["hyperthyroidism"],
    "palpitations": ["hyperthyroidism"],
    "irregular periods": ["PCOS"],
    "acne": ["PCOS"],
    "excess hair growth": ["PCOS"],
}

lab_tests = {
    "fasting blood glucose": {"unit": "mg/dL", "range": (70, 99), "condition": "diabetes"},
    "hba1c": {"unit": "%", "range": (4.0, 5.6), "condition": "diabetes"},
    "tsh": {"unit": "mIU/L", "range": (0.4, 4.0), "condition": "thyroid"},
    "free t4": {"unit": "ng/dL", "range": (0.8, 1.8), "condition": "thyroid"},
    "testosterone": {"unit": "ng/dL", "range": (15, 70), "condition": "pcos"},
    "lh": {"unit": "mIU/mL", "range": (2, 10), "condition": "pcos"},
    "fsh": {"unit": "mIU/mL", "range": (2, 10), "condition": "pcos"},
}

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

# Main app function
def main():
    st.sidebar.title("Navigation")
    main_choice = st.sidebar.selectbox("Select a category:", ["General AI", "Radiologist"])

    if main_choice == "General AI":
        st.header("Welcome to the AI Physician! (Informational)")
        st.warning("""
        Disclaimer: This chatbot is for informational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment.
        It provides basic information based on common symptoms and generalized lab ranges.
        **Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition or the interpretation of your lab results.**
        If you are experiencing a medical emergency, call your local emergency number immediately.
        """)

        st.write("Describe your symptoms or provide lab results, or use the sidebar to access specific models.")

        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'extracted_symptoms' not in st.session_state:
            st.session_state.extracted_symptoms = []
        if 'extracted_lab_results' not in st.session_state:
            st.session_state.extracted_lab_results = {}

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Describe your symptoms or provide lab results (e.g., 'I have a fever and cough', 'my fasting glucose is 110 mg/dL', 'TSH 5.5')")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Use Hugging Face for sentiment analysis
            try:
                hf_result = hf_classifier(user_input)[0]
                label = hf_result['label']
                score = hf_result['score']
            except:
                label = "N/A"
                score = 0

            # Extract symptoms (simple keyword matching)
            extracted_symptoms = []
            user_text_lower = user_input.lower()
            for symptom in symptoms_list:
                if symptom in user_text_lower:
                    extracted_symptoms.append(symptom)
            st.session_state.extracted_symptoms = extracted_symptoms

            # Extract lab results (simple pattern matching)
            # Since no spaCy, we'll do basic parsing (can enhance with regex if needed)
            extracted_lab_results = {}
            for test_name in lab_tests:
                if test_name in user_text_lower:
                    # Try to find a number after the test name
                    start_idx = user_text_lower.find(test_name)
                    snippet = user_input[start_idx:]
                    for word in snippet.split():
                        try:
                            val = float(word)
                            extracted_lab_results[test_name] = {
                                "value": val,
                                "unit": lab_tests[test_name]["unit"],
                                "range": lab_tests[test_name]["range"],
                                "condition": lab_tests[test_name]["condition"]
                            }
                            break
                        except:
                            continue
            st.session_state.extracted_lab_results = extracted_lab_results

            # Generate response
            response_parts = []

            # Only include confidence score
            response_parts.append(f"Confidence: {score:.2f}")

            # Symptoms
            if extracted_symptoms:
                response_parts.append(
                    "Based on your description, I identified the following potential symptoms: " + ", ".join(extracted_symptoms) + "."
                )
                potential_conditions = set()
                for symptom in extracted_symptoms:
                    if symptom in symptom_condition_mapping:
                        potential_conditions.update(symptom_condition_mapping[symptom])
                if potential_conditions:
                    response_parts.append("These symptoms could be associated with: " + ", ".join(potential_conditions) + ".")
                else:
                    response_parts.append("I don't have specific information about conditions linked to these symptoms.")

            # Lab results
            if extracted_lab_results:
                response_parts.append("\nBased on the lab results you provided:")
                for test_name, info in extracted_lab_results.items():
                    val = info["value"]
                    unit = info["unit"]
                    low, high = info["range"]
                    condition_area = info["condition"]
                    interpretation = "within the typical range"
                    if val < low:
                        interpretation = "below the typical range"
                    elif val > high:
                        interpretation = "above the typical range"
                    response_parts.append(
                        f"- Your **{test_name.capitalize()}** is **{val} {unit}**, which is **{interpretation}** for a general range. This test is relevant to **{condition_area.capitalize()}**."
                    )
                response_parts.append("\n**Important Note on Lab Results:**")
                response_parts.append("The reference ranges I use are general. Your lab report will have specific ranges that may differ. The interpretation of lab results is complex and depends on many factors, including your individual health history and other lab values.")
                response_parts.append("**Only a healthcare professional can accurately interpret your lab results in the context of your overall health.**")

            if not extracted_symptoms and not extracted_lab_results:
                response_parts.append("Thank you for sharing. I couldn't identify any common symptoms or relevant lab results from your description.")

            response = "\n\n".join(response_parts)

            # Final disclaimers
            if extracted_symptoms or extracted_lab_results:
                response += "\n\n"
                if extracted_symptoms and extracted_lab_results:
                    response += "**Remember, this analysis is based on limited information and general knowledge. It is NOT a diagnosis.**"
                elif extracted_symptoms:
                    response += "**Remember, this symptom analysis is based on limited information. It is NOT a diagnosis.**"
                elif extracted_lab_results:
                    response += "**Remember, this lab analysis is based on limited information and general ranges. It is NOT a diagnosis.**"
                response += "\n**Always consult a healthcare professional for proper evaluation, diagnosis, and treatment.**"

            # Append assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    elif main_choice == "Radiologist":
        st.header("Radiologist: Image Analysis")
        model_type = st.selectbox("Choose the analysis type:", ["Pneumonia X-ray Analysis"])

        if model_type == "Pneumonia X-ray Analysis":
            st.subheader("Pneumonia X-ray Analysis")
            st.write("Upload a chest X-ray image to get a prediction for pneumonia.")
            uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                if st.button("Analyze X-ray"):
                    with st.spinner("Analyzing image... Please wait."):
                        try:
                            predicted_label, confidence = predict_pneumonia(uploaded_file)
                            if predicted_label == 'Pneumonia':
                                st.markdown(f"<h3 style='color: red;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color: green;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
                            st.write(f"Confidence: {confidence * 100:.2f}%")
                            st.write("Note: This analysis is based on a machine learning model and should not replace a diagnosis from a qualified medical professional.")
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
                            st.write("Please try uploading a valid image file.")

# Run the app
if __name__ == "__main__":
    main()
