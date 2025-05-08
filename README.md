# AIMAIC - AI for Medical Analysis and Image Classification

## Overview

AIMAIC is an interactive web application built with Streamlit that provides medical insights through two main functionalities:

- **General AI (Informational Mode):** A conversational interface where users can describe symptoms or provide lab results, receiving informative feedback, symptom analysis, and general health guidance.
- **Radiologist Mode:** Upload chest X-ray images to receive pneumonia predictions based on a machine learning model.

**Note:** This tool is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment.

---

## Features

- Natural language processing for symptom and lab result analysis
- Sentiment analysis (confidence score only) of user inputs
- Symptom-to-condition mapping and basic lab result interpretation
- Upload and analyze chest X-ray images for pneumonia detection
- Interactive chat interface for easy communication

---

## Technologies Used

- **Streamlit**: For building the web interface
- **TensorFlow**: Loading and running the pneumonia detection model
- **Transformers (Hugging Face)**: Sentiment analysis pipeline
- **NumPy & Keras**: Image preprocessing
- **Python**: Core programming language

---
