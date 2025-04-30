import pandas as pd
import joblib
import numpy as np
import streamlit as st

# Load the trained models
DIABETES_MODEL_FILENAME = 'best_diabetes_rf.pkl'
PCOS_MODEL_FILENAME = 'best_pcos_model_xgboost.pkl'
THYROID_MODEL_FILENAME = 'best_logistic_thyroid_model.pkl'

diabetes_model = joblib.load(DIABETES_MODEL_FILENAME)
pcos_model = joblib.load(PCOS_MODEL_FILENAME)
thyroid_model = joblib.load(THYROID_MODEL_FILENAME)

# Prediction functions
def predict_thyroid(T3_Resin_Uptake_Percentage, Total_Serum_Thyroxine_Isotopic, Total_Serum_Triiodothyronine_Radioimmunoassay, Basal_TSH_Radioimmunoassay, Max_Absolute_Diff_TSH_TRH_Injection):
    custom_input = pd.DataFrame({
        'T3_Resin_Uptake_Percentage': [T3_Resin_Uptake_Percentage],
        'Total_Serum_Thyroxine_Isotopic': [Total_Serum_Thyroxine_Isotopic],
        'Total_Serum_Triiodothyronine_Radioimmunoassay': [Total_Serum_Triiodothyronine_Radioimmunoassay],
        'Basal_TSH_Radioimmunoassay': [Basal_TSH_Radioimmunoassay],
        'Max_Absolute_Diff_TSH_TRH_Injection': [Max_Absolute_Diff_TSH_TRH_Injection]
    })
    predictions = thyroid_model.predict(custom_input)
    probabilities = thyroid_model.predict_proba(custom_input)
    return predictions[0], probabilities[0]

def predict_diabetes(hba1c, glucose):
    custom_input = pd.DataFrame({
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [glucose]
    })
    predictions = diabetes_model.predict(custom_input)
    probabilities = diabetes_model.predict_proba(custom_input)
    return predictions[0], probabilities[0]

def predict_pcos(age, bmi, cycle_ir, pregnant, fsh, lh, lhfshr, hip, waist, whr, tsh, amh, prl, rbs, weight_gain, hair_growth, skin_darkening, pimples, fast_food, l_follicle_count, r_follicle_count, endometrium_mm):
    custom_input = pd.DataFrame({
        'Age (yrs)': [age],
        'BMI': [bmi],
        'Cycle(R/I)': [cycle_ir],
        'Pregnant(Y/N)': [pregnant],
        'FSH(mIU/mL)': [fsh],
        'LH(mIU/mL)': [lh],
        'LHFSHR': [lhfshr],
        'Hip(inch)': [hip],
        'Waist(inch)': [waist],
        'WHR': [whr],
        'TSH': [tsh],
        'AMH': [amh],
        'PRL': [prl],
        'RBS': [rbs],
        'Weight gain': [weight_gain],
        'Hair Growth': [hair_growth],
        'Skin Darkening': [skin_darkening],
        'Pimples': [pimples],
        'Fast food': [fast_food],
        'L Follicle Count': [l_follicle_count],
        'R Follicle Count': [r_follicle_count],
        'Endometrium (mm)': [endometrium_mm]
    })
    predictions = pcos_model.predict(custom_input)
    probabilities = pcos_model.predict_proba(custom_input)
    return predictions[0], probabilities[0]

# Streamlit app code
def main():
   # Main sidebar for model selection
    main_choice = st.sidebar.selectbox("Select a category:", ["Home", "Endocrinologist"])

    if main_choice == "Home":
        st.header("Welcome to AI Physician!")
        st.write("This app utilizes advanced learning algorithms to evaluate your health risk factors based on your inputs. ")
        st.write("Please select a specialist from the sidebar to get started.")
        st.write("### Available Specialists:")
        st.write("##### Endocrinologist:")
        st.write("- **Diabetes Risk Prediction**: Risk prediction based on HbA1c and blood glucose levels.")
        st.write("- **PCOS Risk Prediction**: Prediction based on various health parameters.")
        st.write("- **Thyroid Disease Prediction**: Evaluation based on thyroid function test results.")
        st.write("Please remember, this tool is for informational purposes only. For an accurate diagnosis and personalized medical advice, consult a qualified healthcare professional.")

    elif main_choice == "Endocrinologist":
        # Sub-selection for specific model
        model_type = st.sidebar.selectbox("Choose the health condition:", ["Diabetes", "PCOS", "Thyroid"])
        
        if model_type == "Diabetes":
            st.write("Please enter the following information to predict diabetes risk:")
            hba1c = st.number_input("Enter your HbA1c level:", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            glucose = st.number_input("Enter your blood glucose level:", min_value=0.0, max_value=1000.0, value=140.0, step=1.0)
            if st.button("Predict Diabetes"):
                prediction, probabilities = predict_diabetes(hba1c, glucose)
                confidence = probabilities[prediction]
                if prediction == 1:
                    st.markdown("<h3 style='color: red;'>You are at risk of diabetes!</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: green;'>You are not at risk of diabetes.</h3>", unsafe_allow_html=True)
                st.write(f"Confidence: {confidence * 100:.2f}%")
                st.write("Note: This prediction is based on machine learning models trained on specific data. Consult a healthcare provider for a proper diagnosis.")

        elif model_type == "PCOS":
            st.write("Please enter the following information to predict PCOS risk:")
            age = st.number_input("Age:", min_value=10, max_value=100, value=30, step=1)
            bmi = st.number_input("BMI:", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
            cycle_ir = st.checkbox("Irregular Cycle", value=False)
            pregnant = st.checkbox("Currently Pregnant", value=False)
            fsh = st.number_input("FSH (mIU/mL):", min_value=0.0, max_value=1000.0, value=14.0, step=0.1)
            lh = st.number_input("LH (mIU/mL):", min_value=0.0, max_value=1000.0, value=7.0, step=0.1)
            lhfshr = lh / fsh if fsh != 0 else 0
            hip = st.number_input("Hip (inch):", min_value=0.0, max_value=1000.0, value=38.0, step=0.1)
            waist = st.number_input("Waist (inch):", min_value=0.0, max_value=1000.0, value=33.0, step=0.1)
            whr = waist / hip if hip != 0 else 0
            tsh = st.number_input("TSH:", min_value=0.0, max_value=1000.0, value=3.0, step=0.1)
            amh = st.number_input("AMH:", min_value=0.0, max_value=1000.0, value=5.0, step=0.1)
            prl = st.number_input("Prolactin (PRL):", min_value=0.0, max_value=1000.0, value=25.0, step=0.1)
            rbs = st.number_input("Random Blood Sugar (RBS):", min_value=0.0, max_value=1000.0, value=100.0, step=0.1)
            weight_gain = st.checkbox("Weight Gain", value=False)
            hair_growth = st.checkbox("Hair Growth", value=False)
            skin_darkening = st.checkbox("Skin Darkening", value=False)
            pimples = st.checkbox("Pimples", value=False)
            fast_food = st.checkbox("Fast Food Consumption", value=False)
            l_follicle_count = st.number_input("L Follicle Count:", min_value=0, max_value=1000, value=6, step=1)
            r_follicle_count = st.number_input("R Follicle Count:", min_value=0, max_value=100, value=6, step=1)
            endometrium_mm = st.number_input("Endometrium (mm):", min_value=0, max_value=100, value=8, step=1)
            if st.button("Predict PCOS"):
                prediction, probabilities = predict_pcos(
                    age, bmi, int(cycle_ir), int(pregnant),
                    fsh, lh, lhfshr, hip, waist, whr,
                    tsh, amh, prl, rbs, int(weight_gain),
                    int(hair_growth), int(skin_darkening),
                    int(pimples), int(fast_food),
                    l_follicle_count, r_follicle_count, endometrium_mm
                )
                confidence = probabilities[prediction]
                if prediction == 1:
                    st.markdown("<h3 style='color: red;'>You are at risk of PCOS!</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: green;'>You are not at risk of PCOS.</h3>", unsafe_allow_html=True)
                st.write(f"Confidence: {confidence * 100:.2f}%")
                st.write("Note: This prediction is based on machine learning models trained on specific data. Consult a healthcare provider for a proper diagnosis.")

        elif model_type == "Thyroid":
            st.write("Please enter the following information to predict thyroid disease risk:")
            T3_Resin_Uptake_Percentage = st.number_input("T3 Resin Uptake Percentage:", min_value=0.0, max_value=1000.0, value=100.0, step=0.1)
            Total_Serum_Thyroxine_Isotopic = st.number_input("Total Serum Thyroxine Isotopic:", min_value=0.0, max_value=1000.0, value=10.0, step=0.1)
            Total_Serum_Triiodothyronine_Radioimmunoassay = st.number_input("Total Serum Triiodothyronine Radioimmunoassay:", min_value=0.0, max_value=1000.0, value=2.0, step=0.1)
            Basal_TSH_Radioimmunoassay = st.number_input("Basal TSH Radioimmunoassay", min_value=0.0, max_value=1000.0, value=3.0, step=0.1)
            Max_Absolute_Diff_TSH_TRH_Injection = st.number_input("Max Absolute Diff TSH TRH Injection", min_value=0.0, max_value=1000.0, value=4.0, step=0.1)
            if st.button("Predict"):
                prediction, probabilities = predict_thyroid(
                    T3_Resin_Uptake_Percentage,
                    Total_Serum_Thyroxine_Isotopic,
                    Total_Serum_Triiodothyronine_Radioimmunoassay,
                    Basal_TSH_Radioimmunoassay,
                    Max_Absolute_Diff_TSH_TRH_Injection
                )
                confidence = np.max(probabilities)
                predicted_class_label = prediction
                if prediction == 1:
                    st.markdown("<h3 style='color: green;'>You are not at risk of thyroid disease.</h3>", unsafe_allow_html=True)
                elif prediction == 2:
                    st.markdown("<h3 style='color: red;'>You are at risk of Hypothyroidism!</h3>", unsafe_allow_html=True)
                elif prediction == 3:
                    st.markdown("<h3 style='color: red;'>You are at risk of Hyperthyroidism!</h3>", unsafe_allow_html=True)
                st.write(f"Confidence: {confidence * 100:.2f}%")
                st.write("Note: This prediction is based on machine learning models trained on specific data. Consult a healthcare provider for a proper diagnosis.")

# Run the app
if __name__ == "__main__":
    main()
