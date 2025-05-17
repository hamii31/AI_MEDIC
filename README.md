# AIMAIC - AI for Medical Analysis and Image Classification

## Overview

AIMAIC is an interactive web application built with Streamlit that provides medical insights through two main functionalities:

- **NLP Module:** An NLP-based medical chatbot that extracts symptoms and lab results from user text, interprets medical values against reference ranges, and provides health insights with disclaimers.
- **Radiologist Module:** Upload chest X-ray images to receive pneumonia predictions based on a machine learning model.

**Note:** This tool is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment.

---

## Features
Natural language processing for symptom and lab result extraction and analysis
Basic confidence scoring (sentiment-like) for user inputs
Symptom-to-condition mapping and lab result interpretation based on reference ranges
Upload and analyze chest X-ray images for pneumonia detection
Interactive chat interface for seamless user communication

---

## Technologies Used

- **Streamlit**: For building the web interface
- **TensorFlow**: Loading and running the pneumonia detection model
- **NumPy & Keras**: Image preprocessing
- **Python**: Core programming language

---

## Additional Notes

- **Pneumonia CNN Model**: The original convolutional neural network achieved high accuracy but was too large for practical deployment. To address this, I developed a lightweight version of the model. Initially, the performance of the lightweight model was poor, but I optimized it using techniques such as learning rate scheduling, model checkpointing, weight reuse, architectural adjustments (e.g., adding deeper layers), and batch training. These improvements brought the lightweight model's performance close to that of the original.

- **NLP Module**: I integrated this component to overcome limitations in my earlier tabular models for disease prediction (e.g., Diabetes, Hypo-/Hyperthyroidism, and PCOS). These models relied heavily on structured lab data, which users might not always have. To address this gap, I introduced a Natural Language Processing module that maps free-text symptom and lab report descriptions to likely diagnoses, improving accessibility and model utility.
















ChatGPT can make m
