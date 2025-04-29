import pandas as pd
import joblib
import streamlit as st

# Load the trained models
DIABETES_MODEL_FILENAME = 'best_diabetes_rf.pkl'
PCOS_MODEL_FILENAME = 'best_pcos_model_xgboost.pkl'

diabetes_model = joblib.load(DIABETES_MODEL_FILENAME)
pcos_model = joblib.load(PCOS_MODEL_FILENAME)

# Prediction functions 
def predict_diabetes(hba1c, glucose):
    custom_input = pd.DataFrame({
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [glucose]
    })
    predictions = diabetes_model.predict(custom_input)
    return predictions[0]

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
    return predictions[0]

# Streamlit app code
def main():
    st.title("AI Medical Prediction")
    
    model_selection = st.radio("Choose what you want to predict:", ("Diabetes", "PCOS"))
    
    if model_selection == "Diabetes":
        st.write("Please enter the following information to predict diabetes risk:")
        
        hba1c = st.number_input("Enter your HbA1c level:", min_value=4.0, max_value=15.0, value=5.7, step=0.1)
        glucose = st.number_input("Enter your blood glucose level:", min_value=70, max_value=400, value=120, step=1)
        
        if st.button("Predict Diabetes"):
            prediction = predict_diabetes(hba1c, glucose)
            if prediction == 1:
                st.markdown("<h3 style='color: red;'>You are at risk of diabetes!</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: green;'>You are not at risk of diabetes.</h3>", unsafe_allow_html=True)
    
    elif model_selection == "PCOS":
        st.write("Please enter the following information to predict PCOS risk:")
        
        age = st.number_input("Age:", min_value=10, max_value=100, value=25)
        bmi = st.number_input("BMI:", min_value=10.0, max_value=50.0, value=25.0)
        cycle_ir = st.checkbox("Irregular Cycle", value=False)
        pregnant = st.checkbox("Currently Pregnant", value=False)
        
        fsh = st.number_input("FSH (mIU/mL):", min_value=0.0, max_value=100.0, value=5.0)
        lh = st.number_input("LH (mIU/mL):", min_value=0.0, max_value=100.0, value=5.0)
        lhfshr = lh / fsh if fsh != 0 else 0  # LH/FSH Ratio calculation
        hip = st.number_input("Hip (inch):", min_value=0.0, value=36.0)
        waist = st.number_input("Waist (inch):", min_value=0.0, value=30.0)
        whr = waist / hip if hip != 0 else 0  # Waist-Hip Ratio calculation
        tsh = st.number_input("TSH:", min_value=0.0, max_value=10.0, value=1.0)
        amh = st.number_input("AMH:", min_value=0.0, max_value=10.0, value=1.0)
        prl = st.number_input("Prolactin (PRL):", min_value=0.0, max_value=100.0, value=5.0)
        rbs = st.number_input("RBS:", min_value=0.0, max_value=300.0, value=100.0)
        
        # Checkboxes for symptoms
        weight_gain = st.checkbox("Weight Gain", value=False)
        hair_growth = st.checkbox("Hair Growth", value=False)
        skin_darkening = st.checkbox("Skin Darkening", value=False)
        pimples = st.checkbox("Pimples", value=False)
        fast_food = st.checkbox("Fast Food Consumption", value=False)
        
        # Follicle counts and endometrium
        l_follicle_count = st.number_input("L Follicle Count:", min_value=0, max_value=100, value=10)
        r_follicle_count = st.number_input("R Follicle Count:", min_value=0, max_value=100, value=10)
        endometrium_mm = st.number_input("Endometrium (mm):", min_value=0, max_value=20, value=8)
        
        if st.button("Predict PCOS"):
            prediction = predict_pcos(
                age, bmi, int(cycle_ir), int(pregnant), 
                fsh, lh, lhfshr, hip, waist, whr, 
                tsh, amh, prl, rbs, int(weight_gain), 
                int(hair_growth), int(skin_darkening), 
                int(pimples), int(fast_food), 
                l_follicle_count, r_follicle_count, endometrium_mm
            )
            if prediction == 1:
                st.markdown("<h3 style='color: red;'>You are at risk of PCOS!</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: green;'>You are not at risk of PCOS.</h3>", unsafe_allow_html=True)

    st.write("Note: This prediction is based on machine learning models trained on specific data. Consult a healthcare provider for a proper diagnosis.")

if __name__ == "__main__":
    main()
