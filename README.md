# AIMAIC - AI for Medical Analysis and Image Classification

AIMAIC is an interactive web application built with Streamlit that provides medical insights through two main functionalities:

    NLP Module: A medical chatbot that utilizes a RNN for NLP to interpret user input and return the proper prediction, alongside the confidence it has for the said prediction.

    Radiologist Module: Allows users to upload chest X-ray images to receive pneumonia predictions based on a lightweight, optimized convolutional neural network model.

    Disclaimer: This tool is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment.

    

## Features

    Tokenization of user input and label encoding.

    Recurrent neural network for Natural Language Processing.

    Convolutional neural networks for X-ray Classification.

    Interactive chat interface and image upload functionality.

    

## Technologies Used

    Streamlit – Web interface framework for rapid app development.

    TensorFlow – For loading, building, training and evaluation of various models.

    NumPy & Keras – Image preprocessing and model development.

    Python – Core programming language.

    Pickle - To load the tokenizer and label encoder.


## Additional Notes

- **Pneumonia CNN Model: The original CNN model delivered high accuracy but was too large for practical deployment. To address this, a lightweight version was developed and optimized using learning rate scheduling, model checkpointing, weight reuse, architectural tweaks (including deeper layers), and batch training—resulting in comparable performance to the original. The resulting model outperforms its heavier version, while being 28 times lighter.**

    **Pneumonia CNN Classification Report**
    
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

- **Mammography CNN Model: The model shares the same architecture as the Pneumonia CNN Model.**

    **Mammography CNN Classification Report**
    
    | Class     | Precision | Recall | F1-Score | Support |
    |-----------|-----------|--------|----------|---------|
    | Benign Masses    | 0.95      | 0.92   | 0.93     | 1087     |
    | Malignant Masses   | 0.94      | 0.96   | 0.95     | 1371     |
    | **Accuracy** |        |        | **0.94** | 2458    |
    | Macro Avg | 0.94      | 0.94   | 0.94     | 2458    |
    | Weighted Avg | 0.94   | 0.94   | 0.94     | 2458    |
    
    **Classification Report Summary**
    
    The classification report highlights the excellent performance of this convolutional neural network in differentiating between benign and malignant breast masses. Achieving an overall accuracy of 94%, the model demonstrates high precision (94–95%) and strong recall (92–96%) across both classes, indicating reliable predictive capability in a clinical diagnostic setting.
    
    Notably, the high recall of 0.96 for malignant masses reflects the model's effectiveness in correctly identifying cases of potential concern. This minimizes the risk of false negatives, which is critical in early detection and treatment planning for breast cancer. With balanced performance across both classes, the model is well-suited for deployment in decision support tools aimed at assisting radiologists in breast cancer screening.

- **RNN for NLP Model: Designed to overcome limitations in earlier tabular disease prediction models dependent on structured lab data (e.g., for Diabetes, Thyroid disorders, and PCOS). This RNN utilizes an Embedding layer that turns tokenized input into dense vectors. It is followed by two Bidirectional LSTM layers that process sequences both forwards and backwards. There are multiple Dropout layers to prevent overfitting (since the data, that the model is trained on, is not much). There is a FCL running a 'relu' activation, that connects with the final Dense Layer, which is running a 'softmax' activation.**
- 
| Class     | Precision | Recall | F1-Score | Support |
    |-----------|-----------|--------|----------|---------|
    | Diabetes    | 0.65      | 0.70   | 0.67     | 60     |
    | Hyperthyroidism   | 0.82      | 0.74   | 0.78     | 87     |
    | Hypothyroidism   | 0.67      | 0.63   | 0.65     | 59     |
    | PCOS   | 0.64      | 0.73   | 0.68     | 62    |
    | **Accuracy** |        |        | **0.70** | 268    |
    | Macro Avg | 0.70      | 0.70   | 0.69     | 268    |
    | Weighted Avg | 0.71   | 0.70   | 0.70     | 268    |

    **Classification Report Summary**
  
    The classification report illustrates the performance of the model across multiple medical conditions, achieving an overall accuracy of 70%. The precision scores range from 0.64 to 0.82, indicating moderate to high correctness in positive predictions for each condition. Recall values between 0.63 and 0.74 suggest the model is reasonably effective at identifying actual cases, though there is room for improvement, particularly for hypothyroidism.

    The F1-scores, which balance precision and recall, vary from 0.65 to 0.78, reflecting consistent but not exceptional performance across the different classes. The model demonstrates balanced performance with macro and weighted averages around 0.70, indicating relatively even classification capability among the conditions.

    Overall, while the model shows promising results in distinguishing between various endocrine disorders, further refinement could enhance its sensitivity and specificity, making it more reliable for clinical decision support. This refinement would likely take form of an increase in the dataset volume, since as of now it's incredibly small.


## Sources

- **All datasets, image data, recommendations, and diagnoses used in this project are obtained from verified and reputable sources to ensure accuracy, reliability, and validity. These sources have been thoroughly vetted and are recognized within the relevant scientific and medical communities. The Pneumonia Dataset is from [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) and has been evaluated by three experts. The Mammography Dataset is a dataset cluster created from three datasets - [INbreast Dataset](https://paperswithcode.com/dataset/inbreast), [MIAS Dataset](https://www.kaggle.com/datasets/kmader/mias-mammography) and [DDSM Dataset](http://www.eng.usf.edu/cvprg/Mammography/Database.html). I'm building the dataset for the RNN using BioPortal - created by Stanford Center for Biomedical Informatics Research, in order to find related and reliable articles describing the symptomatics of the diseases, that my model is classifying.**



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
