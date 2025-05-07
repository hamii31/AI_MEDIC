import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import spacy


PNEUMONIA_MODEL_FILENAME = r"lightweight_pneumonia_cnn_512.keras"

# Load the models
try:
    pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_FILENAME)
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}. Make sure the model files exist at the specified paths.")
    st.stop() # Stop the app if models can't be loaded
except Exception as e:
    st.error(f"An error occurred while loading a model: {e}")
    st.stop()
 

# Function to download spaCy model
def download_spacy_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        st.warning(f"SpaCy model '{model_name}' not found. Downloading...")
        try:
            subprocess.check_call(["python", "-m", "spacy", "download", model_name])
            st.success(f"Successfully downloaded spaCy model '{model_name}'. Please refresh the app.")
            st.stop() # Stop the app so Streamlit Cloud can restart with the downloaded model
        except subprocess.CalledProcessError as e:
            st.error(f"Error downloading spaCy model '{model_name}': {e}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during spaCy model download: {e}")
            st.stop()

# Load the small English spaCy model
# This happens only once when the app starts
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download_spacy_model("en_core_web_sm")
except Exception as e:
    st.error(f"An error occurred while loading the spaCy model: {e}")
    st.stop()

# --- Medical Knowledge Base for Chatbot ---
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
    "sore throat": ["common cold", "strep throat"],
    "headache": ["tension headache", "migraine", "flu"],
    "shortness of breath": ["asthma", "pneumonia", "anxiety"],
    "chest pain": ["heart attack (seek immediate medical help!)", "anxiety", "muscle strain"],
    "increased thirst": ["diabetes"],
    "frequent urination": ["diabetes"],
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

    # The model outputs a probability for the 'opacity' class (assuming binary classification: normal=0, opacity=1)
    predicted_class_index = (pred_prob > 0.5).astype(int)

    class_labels = ['Normal', 'Pneumonia']
    predicted_label = class_labels[predicted_class_index]

    # Confidence for the predicted class
    confidence = pred_prob if predicted_class_index == 1 else (1 - pred_prob)

    return predicted_label, confidence, img # Return the image object for display

# Streamlit app code
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

        # --- Chatbot Logic ---
        # Initialize chatbot session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'extracted_symptoms' not in st.session_state:
            st.session_state.extracted_symptoms = []
        if 'extracted_lab_results' not in st.session_state:
            st.session_state.extracted_lab_results = {}


        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input for chatbot
        user_input = st.chat_input("Describe your symptoms or provide lab results (e.g., 'I have a fever and cough', 'my fasting glucose is 110 mg/dL', 'TSH 5.5')")

        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # --- NLP Processing with spaCy ---
            doc = nlp(user_input)

            # --- Symptom Extraction (Using spaCy and Keyword Matching) ---
            extracted_symptoms = []
            user_text_lower = user_input.lower()

            for symptom in symptoms_list:
                if symptom in user_text_lower:
                    extracted_symptoms.append(symptom)

            st.session_state.extracted_symptoms = extracted_symptoms

            # --- Lab Result Extraction (Using spaCy and Pattern Matching) ---
            extracted_lab_results = {}

            for i, token in enumerate(doc):
                token_text_lower = token.text.lower()
                for test_name, test_info in lab_tests.items():
                    if test_name in token_text_lower:
                        for j in range(i + 1, min(i + 5, len(doc))):
                            next_token = doc[j]
                            if next_token.like_num:
                                try:
                                    value = float(next_token.text)
                                    extracted_lab_results[test_name] = {"value": value, "unit": test_info["unit"], "range": test_info["range"], "condition": test_info["condition"]}
                                    break
                                except ValueError:
                                    pass

            st.session_state.extracted_lab_results = extracted_lab_results

            # --- Generate Chatbot Response ---
            response_parts = []

            if extracted_symptoms:
                response_parts.append("Based on your description, I identified the following potential symptoms: " + ", ".join(extracted_symptoms) + ".")

                potential_conditions_from_symptoms = set()
                for symptom in extracted_symptoms:
                    if symptom in symptom_condition_mapping:
                        potential_conditions_from_symptoms.update(symptom_condition_mapping[symptom])

                if potential_conditions_from_symptoms:
                    response_parts.append("These symptoms could be associated with conditions such as: " + ", ".join(list(potential_conditions_from_symptoms)) + ".")
                else:
                    response_parts.append("I don't have specific information about conditions directly linked to these symptoms in my current knowledge base.")

            if extracted_lab_results:
                response_parts.append("\nBased on the lab results you provided:")
                for test_name, result_info in extracted_lab_results.items():
                    value = result_info["value"]
                    unit = result_info["unit"]
                    lower_bound, upper_bound = result_info["range"]
                    condition_area = result_info["condition"]

                    interpretation = "within the typical range"
                    if value < lower_bound:
                        interpretation = "below the typical range"
                    elif value > upper_bound:
                        interpretation = "above the typical range"

                    response_parts.append(f"- Your **{test_name.capitalize()}** is **{value} {unit}**, which is **{interpretation}** for a general range. This test is relevant to **{condition_area.capitalize()}**.")

                response_parts.append("\n**Important Note on Lab Results:**")
                response_parts.append("The reference ranges I use are general. Your lab report will have specific ranges that may differ. The interpretation of lab results is complex and depends on many factors, including your individual health history and other lab values.")
                response_parts.append("**Only a healthcare professional can accurately interpret your lab results in the context of your overall health.**")


            if not extracted_symptoms and not extracted_lab_results:
                response_parts.append("Thank you for sharing. I couldn't identify any common symptoms or relevant lab results from my current knowledge base in your description.")

            response = "\n\n".join(response_parts)

            # Add final disclaimers for the chatbot response
            if extracted_symptoms or extracted_lab_results:
                response += "\n\n"
                if extracted_symptoms and extracted_lab_results:
                     response += "**Remember, this analysis is based on limited information and general knowledge. It is NOT a diagnosis.**"
                elif extracted_symptoms:
                    response += "**Remember, this symptom analysis is based on limited information. It is NOT a diagnosis.**"
                elif extracted_lab_results:
                     response += "**Remember, this lab analysis is based on limited information and general ranges. It is NOT a diagnosis.**"

                response += "\n**Always consult a healthcare professional for proper evaluation, diagnosis, and treatment.**"


            # Add chatbot response to history
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

            if uploaded_file is not None:

                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

                if st.button("Analyze X-ray"):
                    with st.spinner("Analyzing image... Please wait."):
                        try:
                            predicted_label, confidence, img_display = predict_pneumonia(uploaded_file)

                            st.write("### Prediction:")
                            if predicted_label == 'Pneumonia':
                                st.markdown(f"<h3 style='color: red;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color: green;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
                            st.write(f"Confidence: {confidence * 100:.2f}%")

                            st.write("Note: This analysis is based on a machine learning model and should not replace a diagnosis from a qualified medical professional.")

                        except Exception as e:
                            st.error(f"An error occurred during analysis: {e}")
                            st.write("Please try uploading a valid image file.")

# Run the app
if __name__ == "__main__":
    main()
