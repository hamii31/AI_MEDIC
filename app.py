import pandas as pd
import joblib
import streamlit as st

# Load the trained model
DRFM = 'best_diabetes_rf.pkl'
loaded_model = joblib.load(DRFM)

# Create function for making predictions based on user inputs
def predict_diabetes(hba1c, glucose):
    # Prepare the input data as a DataFrame
    custom_input = pd.DataFrame({
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [glucose]
    })
    
    # Make predictions
    predictions = loaded_model.predict(custom_input)
    return predictions[0]  # Return the first prediction (since we only pass one row)

# Streamlit app code
def main():
    st.title("Diabetes Prediction")
    
    st.write("Please enter the following information to predict diabetes risk:")
    
    # User input for HbA1c level
    hba1c = st.number_input("Enter your HbA1c level:", min_value=4.0, max_value=15.0, value=5.7, step=0.1)
    
    # User input for blood glucose level
    glucose = st.number_input("Enter your blood glucose level:", min_value=70, max_value=400, value=120, step=1)
    
    # Predict button
    if st.button("Predict"):
        prediction = predict_diabetes(hba1c, glucose)
        if prediction == 1:
            st.markdown("<h3 style='color: red;'>You are at risk of diabetes!</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>You are not at risk of diabetes.</h3>", unsafe_allow_html=True)
    
    # Optionally, you can provide more information or insights here
    st.write("Note: This prediction is based on a machine learning model trained on specific data. Consult a healthcare provider for a proper diagnosis.")

# Run the app
if __name__ == "__main__":
    main()
