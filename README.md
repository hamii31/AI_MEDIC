# AIMAIC - AI for Medical Analysis and Image Classification

AIMAIC is an interactive web application built with Streamlit that provides medical insights through two main functionalities:

    NLP Module: An intelligent medical chatbot that extracts symptoms and lab results from user input text, interprets medical values against reference ranges, and offers health insights with appropriate disclaimers.

    Radiologist Module: Allows users to upload chest X-ray images to receive pneumonia predictions based on a lightweight, optimized convolutional neural network model.

    Disclaimer: This tool is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment.

    

## Features

    Natural language processing for symptom and lab result extraction and analysis.

    Basic confidence scoring (similar to sentiment analysis) for user inputs.

    Symptom-to-condition mapping and lab result interpretation based on medical reference ranges.

    Upload and analyze chest X-ray images to detect pneumonia.

    Interactive chat interface for seamless communication and health insight delivery.

    

## Technologies Used

    Streamlit – Web interface framework for rapid app development.

    TensorFlow – Loading and running the pneumonia detection model.

    NumPy & Keras – Image preprocessing and model development.

    Python – Core programming language.



## Additional Notes

- **Pneumonia CNN Model: The original CNN model delivered high accuracy but was too large for practical deployment. To address this, a lightweight version was developed and optimized using learning rate scheduling, model checkpointing, weight reuse, architectural tweaks (including deeper layers), and batch training—resulting in comparable performance to the original.**

Pneumonia CNN Classification Report

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Normal    | 0.95      | 0.92   | 0.94     | 468     |
| Opacity   | 0.95      | 0.97   | 0.96     | 780     |
| **Accuracy** |        |        | **0.95** | 1248    |
| Macro Avg | 0.95      | 0.95   | 0.95     | 1248    |
| Weighted Avg | 0.95   | 0.95   | 0.95     | 1248    |

**Classification Report Summary**

The classification report clearly demonstrates the strong capability of this convolutional neural network to accurately distinguish between normal and opacity cases in medical imaging. With an overall accuracy of 95%, high precision (95-96%), and excellent recall (91-97%), the model shows promising reliability suitable for clinical applications.

Importantly, the high recall for the opacity class (0.97) indicates the model’s effectiveness at correctly identifying potential disease cases, minimizing the risk of missed diagnoses.

- **Mammography CNN Model: 

- **NLP Module: Designed to overcome limitations in earlier tabular disease prediction models dependent on structured lab data (e.g., for Diabetes, Thyroid disorders, and PCOS). This module uses free-text input to map symptoms and lab reports to likely diagnoses, making the tool more accessible and useful when structured data is unavailable. Then decideswhich diagnose best matches the provided symtpoms and displays it as the more likely diagnose. If there's a recommendation for that top diagnosis, it will be provided.**


## Sources

- **All datasets, image data, recommendations, and diagnoses used in this project are obtained from verified and reputable sources to ensure accuracy, reliability, and validity. These sources have been thoroughly vetted and are recognized within the relevant scientific and medical communities. The Pneumonia Dataset is from Kaggle and has been evaluated by three experts. The Mammography Dataset is a dataset cluster from 3 additional datasets - INbreast Dataset, MIAS Dataset and DDSM Dataset. I'm building the `disease_db.py` using BioPortal - created by Stanford Center for Biomedical Informatics Research**



## Installation

```
git clone <repository-url>
cd AIMAIC
pip install -r requirements.txt`
```


## Running the App

```
streamlit run app.py
```
