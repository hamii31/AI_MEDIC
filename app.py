import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from transformers import pipeline
import re

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


# Helper function: get_lab_test_details
def get_lab_test_details(test_name, value, preferred_unit=None, lab_tests_dict=None):
    if lab_tests_dict is None:
        lab_tests_dict = lab_tests  # fallback if not provided
    test_info = lab_tests_dict.get(test_name.lower())
    if not test_info:
        return value, "unit not found", "range not available", "interpretation not available"

    # Handle nested units
    if isinstance(test_info, dict) and "units" in test_info:
        units_dict = test_info["units"]
        # Determine which unit to use
        unit_to_use = preferred_unit or test_info.get("preferred_unit") or list(units_dict.keys())[0]
        unit_details = units_dict.get(unit_to_use)
        if not unit_details:
            unit_to_use = list(units_dict.keys())[0]
            unit_details = units_dict[unit_to_use]
        range_vals = unit_details["range"]
        interpretation = unit_details.get("interpretation", "No interpretation available")
        return value, unit_to_use, range_vals, interpretation
    else:
        unit = test_info.get("unit", "unit")
        range_vals = test_info.get("range")
        condition = test_info.get("condition", "No condition info")
        return value, unit, range_vals, condition


# --- Medical Knowledge Base ---
symptoms_list = [
    "fever", "cough", "sore throat", "headache", "fatigue",
    "shortness of breath", "chest pain", "nausea", "vomiting",
    "diarrhea", "abdominal pain", "rash", "dizziness", "loss of taste", "loss of smell",
    "increased thirst", "frequent urination", "unexplained weight loss", "blurred vision",
    "weight gain", "weight loss", "hair loss", "dry skin", "feeling cold", "feeling hot", "palpitations",
    "irregular periods", "acne", "excess hair growth", "weight gain (especially around the middle)",
    "muscle weakness", "constipation", "depression", "irritability", "mood swings",
    "cold intolerance", "hot flashes", "menstrual irregularities", "fertility issues", "irregular painful periods"
]

symptom_condition_mapping = {
    "irregular painful periods": ["PCOS"],
    "fever": ["flu", "common cold", "infection"],
    "cough": ["flu", "common cold", "bronchitis"],
    "sore throat": ["common cold", "strep throat", "sore throat"],
    "headache": ["tension headache", "migraine", "flu"],
    "shortness of breath": ["asthma", "pneumonia", "anxiety"],
    "chest pain": ["heart attack (seek immediate medical help!)", "anxiety", "muscle strain"],
    "increased thirst": ["diabetes"],
    "frequent urination": ["diabetes"],
    "unexplained weight loss": ["diabetes", "hyperthyroidism"],
    "blurred vision": ["diabetes", "hyperthyroidism"],
    "weight gain": ["hypothyroidism", "PCOS"],
    "weight loss": ["hyperthyroidism"],
    "hair loss": ["hypothyroidism", "PCOS"],
    "dry skin": ["hypothyroidism"],
    "feeling cold": ["hypothyroidism"],
    "feeling hot": ["hyperthyroidism"],
    "palpitations": ["hyperthyroidism"],
    "irregular periods": ["PCOS", "thyroid disorders"],
    "acne": ["PCOS", "hormonal imbalance"],
    "excess hair growth": ["PCOS"],
    "muscle weakness": ["hypothyroidism"],
    "constipation": ["hypothyroidism"],
    "depression": ["hypothyroidism", "diabetes"],
    "irritability": ["hyperthyroidism", "diabetes"],
    "mood swings": ["hyperthyroidism", "PCOS"],
    "cold intolerance": ["hypothyroidism"],
    "hot flashes": ["menopause", "hyperthyroidism"],
    "menstrual irregularities": ["PCOS", "thyroid disorders"],
    "fertility issues": ["PCOS", "thyroid disorders"]
}

# Original lab_tests dictionary
lab_tests = {
    "fasting blood glucose": {
        "unit": "mg/dL",
        "range": (70, 99),
        "condition": "Normal; ≥126 mg/dL indicates diabetes",
        "gender": "both"
    },
    "hba1c": {
        "unit": "%",
        "range": (4.0, 5.6),
        "condition": "Normal; ≥6.5% indicates diabetes",
        "gender": "both"
    },
    "glucose tolerance test": {
        "unit": "mg/dL",
        "range": "less than 140 mg/dL (normal), 140-199 mg/dL (pre-diabetes), ≥200 mg/dL (diabetes)",
        "condition": "Assessment of glucose metabolism",
        "gender": "both"
    },
    "lipid profile": {
        "unit": "mg/dL",
        "range": {
            "total cholesterol": (125, 200),
            "LDL": ("less than 100"),
            "HDL": ("greater than 40")
        },
        "condition": "Cardiovascular risk assessment",
        "gender": "both"
    },
    "tsh": {
        "unit": "mIU/L",
        "range": (0.4, 4.0),
        "condition": "Normal; <0.4 (hyperthyroidism), >4.0 (hypothyroidism)",
        "gender": "both"
    },
    "ft4": {
        "units": {
            "ng/dL": {
                "range": (0.8, 1.8),
                "interpretation": "Normal; low in Hypothyroidism or Pituitary disorders, high in Hyperthyroidism and Thyroiditis"
            },
            "pmol/L": {
                "range": (12, 23),
                "interpretation": "Normal; low in Hypothyroidism or Pituitary disorders, high in Hyperthyroidism and Thyroiditis"
            }
        },
        "condition": "Free T4; low in Hypothyroidism or Pituitary disorders, high in Hyperthyroidism and Thyroiditis",
        "preferred_unit": "pmol/L",
        "gender": "both"
    },
    "ft3": {
        "units": {
            "ng/dL": {
                "range": (2.3, 4.2),
                "interpretation": "Normal; high in Hyperthyroidism and Thyroiditis, low in Hypothyroidism"
            },
            "pmol/L": {
                "range": (3.1, 6.9),
                "interpretation": "Normal; high in Hyperthyroidism and Thyroiditis, low in Hypothyroidism"
            }
        },
        "condition": "Free T3; high in Hyperthyroidism and Thyroiditis, low in Hypothyroidism",
        "preferred_unit": "pmol/L",
        "gender": "both"
    },
    "tt3": {
        "unit": "ng/dL",
        "range": (80, 180),
        "condition": "Normal",
        "gender": "both"
    },
    "amh": {
        "unit": "ng/mL",
        "range": (1.0, 4.0),
        "interpretation": "Normal ovarian reserve; lower in diminished reserve",
        "gender": "female"
    },
    "lh": {
        "units": {
            "pmol/L": {
                "interpretation": "Normal; <1.5 (Hypopituitarism, Secondary ovarian insufficiency), >12.5 (PCOS, POI, Menopause)",
                "range": (1.5, 12.0)
            },
            "IU/L": {
                "interpretation": "Normal;  <1.5 (Hypopituitarism, Secondary ovarian insufficiency), >9.5 (PCOS, POI, Menopause)",
                "range": (1.5, 9.3)
            },
            "ng/mL": {
                "interpretation": "Normal;  <0.1 (Hypopituitarism, Secondary ovarian insufficiency), >1.0 (PCOS, POI, Menopause)",
                "range": (0.1, 1.0)
            },
        },
        "condition": "Luteinizing hormone; low LH may indicate Hypopituitarism, Secondary ovarian insufficiency, Elevated LH may indicate PCOS, POI or Menopause",
        "gender": "female"
    },
    "fsh": {
        "units": {
            "pmol/L": {
                "interpretation": "Normal; <3.5 (Hypopituitarism, Secondary ovarian insufficiency), >12.5 (POI, Menopause, Turner Syndrome)",
                "range": (3.5, 12.5)
            },
            "IU/L": {
                "interpretation": "Normal; <1.5 (Hypopituitarism, Secondary ovarian insufficiency), >12.5 (POI, Menopause, Turner Syndrome)",
                "range": (1.5, 12.5)
            },
            "ng/mL": {
                "interpretation": "Normal; <0.35 (Hypopituitarism, Secondary ovarian insufficiency), >1.25 (POI, Menopause, Turner Syndrome)",
                "range": (0.35, 1.25)
            },
        },
        "condition": "Follicle-stimulating hormone; low FSH may indicate Hypopituitarism, Secondary ovarian insufficiency, high FSH may indicate POI, Menopause, Turner Syndrome",
        "gender": "female"
    },
    "dhea-s": {
        "unit": "μg/dL",
        "range": (65, 380),
        "condition": "Normal; elevated in adrenal hyperplasia or PCOS",
        "gender": "both"
    },
    "testosterone": {
        "unit": "ng/dL",
        "range": (15, 70),
        "condition": "Normal; elevated in PCOS",
        "gender": "both"
    },
    "anti-thyroid antibodies": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": "Negative indicates absence of autoimmune thyroid disease",
        "gender": "both"
    },
    "tat": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": "Negative indicates absence of thyroid autoimmune activity",
        "gender": "both"
    },
    "mat": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": "Negative indicates absence of autoimmune thyroid disease",
        "gender": "both"
    },
    "cortisol": {
        "unit": "μg/dL",
        "range": (6, 23),
        "condition": "Normal morning levels; low suggests adrenal insufficiency, high suggests Cushing's syndrome",
        "gender": "both"
    }
}

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
    main_choice = st.sidebar.selectbox("Select a category:", ["General AI", "Radiologist"])

    if main_choice == "General AI":
        st.header("Welcome to AIMAIC - AI for Medical Analysis and Image Classification")
        st.warning("""
        Disclaimer: This chatbot focuses mostly on thyroid problems, diabetes and PCOS. More diseases will be added in the future.    
        **Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition or the interpretation of your lab results.**
        If you are experiencing a medical emergency, call your local emergency number immediately.
        """)

        st.warning("Usage: Before starting the chat, please select your gender from the sidebar for more accurate results. Describe your symptoms or provide lab results, including the proper units, or use the sidebar to access specific models for image analysis.")

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

            # Use Hugging Face for sentiment analysis
            try:
                hf_result = hf_classifier(user_input)[0]
                score = hf_result['score']
            except:
                score = 0

            # Extract symptoms with exact matching
            extracted_symptoms = []
            user_text_lower = user_input.lower()
            for symptom in symptoms_list:
                pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
                if re.search(pattern, user_text_lower):
                    extracted_symptoms.append(symptom)
            st.session_state.extracted_symptoms = extracted_symptoms

            # Select relevant lab tests based on gender
            if user_gender == "male":
                current_lab_tests = lab_tests_male
            else:
                current_lab_tests = lab_tests_female

            # Extract lab results with flexible patterns
            extracted_lab_results = {}
            for test_name, test_info in current_lab_tests.items():
                pattern_name = re.escape(test_name.lower())
                patterns = [
                    rf'{pattern_name}\s*(?:is|:|=|-)?\s*([0-9.]+)',    # e.g., "FT4 is 15"
                    rf'{pattern_name}\s*[^\w]([0-9.]+)',               # e.g., "FT4 15"
                    rf'[^\w]{pattern_name}\s*[^\w]([0-9.]+)',         # e.g., "free T4 - 15"
                    rf'{pattern_name}\s*[\(\[]?\s*([0-9.]+)',         # e.g., "FT4 (15)"
                ]
                found = False
                for pattern in patterns:
                    match = re.search(pattern, user_text_lower)
                    if match:
                        try:
                            value = float(match.group(1))
                            # Optionally, find nearby units in user input
                            snippet_start = match.end()
                            snippet = user_input[snippet_start:snippet_start+20]
                            unit_found = None
                            # Check if units are specified in snippet
                            for unit_name in (test_info.get('unit') or []):
                                if unit_name.lower() in snippet.lower():
                                    unit_found = unit_name
                                    break
                            if not unit_found:
                                # Default to first available unit
                                unit_found = test_info.get('unit', [None])[0]
                            # Save extracted result
                            range_vals = test_info.get("range")
                            extracted_lab_results[test_name] = {
                                "value": value,
                                "unit": unit_found,
                                "range": range_vals,
                                "condition": test_info.get('condition')
                            }
                            found = True
                            break
                        except:
                            continue
                if found:
                    continue
            st.session_state.extracted_lab_results = extracted_lab_results

            # Generate response
            response_parts = []
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
                    preferred_unit = info.get("unit")
                    # Use the appropriate lab_tests dict
                    if user_gender == "male":
                        test_info = lab_tests_male.get(test_name)
                    else:
                        test_info = lab_tests_female.get(test_name)
                    value, unit, range_vals, interpretation = get_lab_test_details(
                        test_name, val, preferred_unit, lab_tests_dict=(lab_tests_male if user_gender=="male" else lab_tests_female)
                    )

                    # Interpret range
                    if range_vals:
                        if isinstance(range_vals, (tuple, list)):
                            low, high = range_vals
                            if val < low:
                                interpretation = "below the typical range"
                            elif val > high:
                                interpretation = "above the typical range"
                            else:
                                interpretation = "within the typical range"
                        elif isinstance(range_vals, str):
                            interpretation = range_vals
                        else:
                            interpretation = "range info not available"
                    else:
                        interpretation = "range info not available"

                    # Generate message
                    response_parts.append(
                        f"- Your **{test_name.capitalize()}** is **{value} {unit}**, which is **{interpretation}** for a general range. Additional info: **{test_info.get('condition', 'N/A')}**."
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
