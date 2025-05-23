import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from disease_db import disease_db, lab_tests
import re

# Paths and filenames
PNEUMONIA_MODEL_FILENAME = r"lightweight_pneumonia_cnn_512.keras"
MAMMOGRAPHY_MODEL_FILENAME = r"lightweight_mammography_cnn_512.keras"
# Load the CNNs
try:
    pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_FILENAME)
    mammography_model = tf.keras.models.load_model(MAMMOGRAPHY_MODEL_FILENAME)  # Placeholder for breast cancer model
except FileNotFoundError as e:
    st.error(f"Error loading one or multiple models:")
    st.stop()
except Exception as e:
    st.error(f"Error loading a model: {e}")
    st.stop()


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

# Helper function: get_lab_test_details
def get_lab_test_details(test_name, value, lab_tests_dict=None):
    if lab_tests_dict is None:
        lab_tests_dict = lab_tests  # fallback if not provided
    test_info = lab_tests_dict.get(test_name.lower())
    if not test_info:
        return value, "unit not found", "range not available", "interpretation not available"

    unit = test_info.get("unit", "unit")
    range_vals = test_info.get("range")
    condition = test_info.get("condition", "No condition info")

    return value, unit, range_vals, condition


# --- Split lab_tests into male and female based on 'gender' key ---
lab_tests_male = {}
lab_tests_female = {}
for test_name, test_info in lab_tests.items():
    gender_group = test_info.get("gender", "both")
    if gender_group == "both":
        lab_tests_male[test_name] = test_info
        lab_tests_female[test_name] = test_info
    elif gender_group == "male":
        lab_tests_male[test_name] = test_info
    elif gender_group == "female":
        lab_tests_female[test_name] = test_info



# Main app function
def main():
    st.sidebar.title("Navigation")
    main_choice = st.sidebar.selectbox("Select a category:", ["Medical Chatbot", "Radiologist"])

    if main_choice == "Medical Chatbot":
        st.header("Welcome to AIMAIC - AI for Medical Analysis and Image Classification")
        st.warning("""
        Disclaimer: This chatbot focuses mostly on thyroid problems, diabetes and PCOS. More diseases will be added in the future.    .
        **Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition or the interpretation of your lab results.**
        If you are experiencing a medical emergency, call your local emergency number immediately.
        """)

        st.warning("Usage: Before starting the chat, please select your gender for more accurate results. Describe your symptoms or provide lab results, including the proper units, or use the sidebar to access specific models for image analysis.")

        # Select user gender
        user_gender = st.sidebar.selectbox("Select your gender:", ["male", "female"])

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

        user_input = st.chat_input("Start chat (e.g., 'I have a fever and cough', 'my fasting glucose is 110 mg/dL', 'TSH 5.5')")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Extract symptoms
            extracted_symptoms = []
            user_text_lower = user_input.lower()
            for symptom in disease_db:
                pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
                if re.search(pattern, user_text_lower):
                    extracted_symptoms.append(symptom)
            st.session_state.extracted_symptoms = extracted_symptoms

            # Extract lab results
            current_lab_tests = lab_tests_female if user_gender == "female" else lab_tests_male
            extracted_lab_results = {}
            for test_name, info in current_lab_tests.items():
                pattern_name = re.escape(test_name.lower())
                patterns = [
                    rf'{pattern_name}\s*(?:is|:|=|-)?\s*([0-9.]+)',
                    rf'{pattern_name}\s*[^\w]([0-9.]+)',
                    rf'[^\w]{pattern_name}\s*[^\w]([0-9.]+)',
                    rf'{pattern_name}\s*[\(\[]?\s*([0-9.]+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, user_text_lower)
                    if match:
                        try:
                            value = float(match.group(1))
                            snippet_start = match.end()
                            snippet = user_input[snippet_start:snippet_start+20]
                            unit_found = None
                            for unit_name in (info.get('unit') or []):
                                if unit_name.lower() in snippet.lower():
                                    unit_found = unit_name
                                    break
                            if not unit_found:
                                unit_found = info.get('unit', [None])
                            range_vals = info.get("range")
                            extracted_lab_results[test_name] = {
                                "value": value,
                                "unit": unit_found,
                                "range": range_vals,
                                "condition": info.get('condition')
                            }
                            break
                        except:
                            continue
            st.session_state.extracted_lab_results = extracted_lab_results

            # Simple chatbot reply
            response_parts = []

            # Symptoms
            if extracted_symptoms:
                response_parts.append(
                    "I identified the following symptoms: " + ", ".join(extracted_symptoms) + "."
                )
                potential_conditions = set()
                for symptom in extracted_symptoms:
                    if symptom in disease_db:
                        potential_conditions.update(disease_db[symptom])
                if potential_conditions:
                    response_parts.append("These symptoms may be associated with: " + ", ".join(potential_conditions) + ".")
                else:
                    response_parts.append("I could not map the symptoms to known conditions.")

            # Lab Results
            if extracted_lab_results:
                response_parts.append("\nBased on your lab results:")
                for test_name, info in extracted_lab_results.items():
                    val = info["value"]
                    unit = info.get("unit", "")
                    range_vals = info.get("range")
                    condition = info.get("condition", {})

                    if range_vals:
                        low, high = range_vals
                        if val < low:
                            status = "below the normal range"
                            cond_info = condition.get("below", "")
                        elif val > high:
                            status = "above the normal range"
                            cond_info = condition.get("above", "")
                        else:
                            status = "within the normal range"
                            cond_info = condition.get("normal", "")
                    else:
                        status = "range unknown"
                        cond_info = "Reference range is not available for this test."

                    response_parts.append(
                        f"- Your **{test_name}** is **{val} {unit}**, which is **{status}**. Related to: **{cond_info}**."
                    )

                response_parts.append("""
                **Note:** These are general reference ranges. Always refer to your own lab report and discuss with your doctor.
                """.strip())

            if not extracted_symptoms and not extracted_lab_results:
                response_parts.append("Thank you for sharing. I could not identify specific symptoms or lab results. Try rephrasing your input (My TSH is 4, I have a fever, I have a headache).")

            # Final disclaimers
            if extracted_symptoms or extracted_lab_results:
                response_parts.append("""
                **This is not a medical diagnosis. Please consult a healthcare provider for personalized medical advice.**
                """.strip())

            response = "\n\n".join(response_parts)

            # Show response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    elif main_choice == "Radiologist":
        st.header("Radiologist: Image Analysis")
        model_type = st.sidebar.selectbox("Choose the analysis type:", ["Pneumonia X-ray Analysis", "Breast Cancer X-ray Analysis"])

        if model_type == "Pneumonia X-ray Analysis":
            st.subheader("Pneumonia X-ray Analysis")
            st.write("Upload a chest X-ray image to get a prediction for pneumonia.")
            uploaded_file = st.file_uploader("Choose a X-ray image...", type=["jpg", "jpeg", "png"])

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
        elif model_type == "Breast Cancer X-ray Analysis":
            st.subheader("Breast X-ray Analysis")
            st.write("Upload a breast X-ray image to get a prediction for breast cancer. Note: This model is still undergoing development.")
            uploaded_file = st.file_uploader("Choose a X-ray image...", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                if st.button("Analyze X-ray"):
                    with st.spinner("Analyzing image... Please wait."):
                        try:
                            predicted_label, confidence = predict_breast_cancer(uploaded_file)
                            if predicted_label == 'Malignant':
                                st.markdown(f"<h3 style='color: red;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color: green;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
                            st.write(f"Confidence: {confidence * 100:.2f}%")
                            st.write("Note: This analysis is based on a machine learning model and should not replace a diagnosis from a qualified medical professional.")
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
                            st.write("Please try uploading a valid image file.")

if __name__ == "__main__":
    main()
