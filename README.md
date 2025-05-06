## Machine Learning Models

This application utilizes two distinct machine learning models to provide predictions:

### 1. Thyroid Prediction Classifier

*   **Purpose:** Predicts the presence or absence of thyroid conditions based on patient health data.
*   **Algorithm:** Logistic Regression.
*   **Input Data:** Structured data loaded from a CSV file. The features used are all columns and contain various health parameters related to thyroid function.
*   **Output:** A predicted class label representing the thyroid status - underactive, overactive or normal.
*   **Training Data:** Trained on the Thyroid Disease Dataset from UCI Machine Learning Repository ([https://archive.ics.uci.edu/dataset/102/thyroid+disease]). 
*   **Model File:** The trained model is saved as a pickle file named `best_logistic_thyroid_model.pkl`.
*   **Dependencies:** Pandas, Scikit-learn, Joblib.

### 2. Lightweight Pneumonia Detection CNN

*   **Title:** Lightweight PneumoniaCNN
*   **Model Type:** Lightweight Convolutional Neural Network (CNN) for Binary Image Classification.
*   **Purpose:** To classify chest X-ray images as either "normal" or "opacity" (pneumonia), with a focus on reduced model complexity and size compared to standard CNN architectures.
*   **Input Data:** Chest X-ray images, resized to 224x224 pixels with 3 color channels (RGB).
*   **Output:** A binary prediction:
    *   `0` (or a low probability value below the threshold, e.g., 0.5) representing "normal".
    *   `1` (or a high probability value above the threshold, e.g., 0.5) representing "opacity" (pneumonia).
    The model outputs a probability score between 0 and 1, which can be thresholded (e.g., at 0.5) for final classification.
*   **Dataset:** Trained and evaluated on the Chest X-Ray Images (Pneumonia) dataset from Kaggle ([https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)). The dataset is split into training, validation, and testing sets.
*   **Data Augmentation:** The training data is augmented using techniques like rotation, zoom, horizontal flipping, and shifting to improve the model's robustness.
*   **Training:** The model is trained using the Adam optimizer and binary cross-entropy loss. Training includes a learning rate scheduler (`ReduceLROnPlateau`) to adjust the learning rate based on validation accuracy.
*   **Model File:** The lightweight model is saved as a Keras file: `pneumonia_model_lightweight_512.keras`.
*   **Dependencies:** TensorFlow, Keras, NumPy, Scikit-learn 
*   **Usage:** The model was [evaluated](https://youtu.be/YEn74_YTs2Q) using assets from the CheXpert dataset, which were not included in the training data in order to check the generalization of the model.

### 3. PCOS Prediction XGBoost Model

*   **Title:** PCOS Prediction XGBoost Model
*   **Model Type:** eXtreme Gradient Boosting (XGBoost) Classifier.
*   **Purpose:** To predict the likelihood of Polycystic Ovary Syndrome (PCOS) based on a set of physiological and lifestyle features.
*   **Input Data:** Tabular data containing various features related to health metrics, hormonal levels, anthropometric measurements, and lifestyle factors.
*   **Output:** A binary prediction:
    *   `0` representing the absence of PCOS.
    *   `1` representing the presence of PCOS.
*   **Dataset:** The model is trained and evaluated on a custom dataset constructed from PCOS Dataset from Kaggle ([https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos]).
*   **Data Preprocessing:**
    *   Column names are cleaned by removing leading/trailing whitespace.
    *   A specific subset of relevant columns is selected for the model.
    *   Missing numerical values are imputed using the median of the respective columns.
    *   Relevant columns are explicitly converted to a numeric format.
*   **Model Training:**
    *   The dataset is split into training and testing sets (70% train, 30% test).
    *   An XGBoost Classifier is used as the base model.
    *   **Class Weighting:** The model is trained with class weights to address potential class imbalance, giving more importance to the positive class (PCOS). In this case, the positive class is weighted 5 times more than the negative class (normal).
*   **Evaluation:**
    *   The model is evaluated on the held-out test set.
    *   **Accuracy:** The overall accuracy of the model on the test set is reported.
    *   **Confusion Matrix:** A normalized confusion matrix is generated and visualized to show the proportion of true positive, true negative, false positive, and false negative predictions.
    *   **Feature Importance:** The script calculates and visualizes the importance of each feature in the model's decision-making process.
*   **Model File:** The trained XGBoost model is saved to a pickle file (`best_pcos_model_xgboost.pkl`) using `joblib`.
*   **Dependencies:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, xgboost.

### 4. Diabetes Prediction Random Forest Model

*   **Title:** Diabetes Prediction Random Forest Model
*   **Model Type:** Random Forest Classifier.
*   **Purpose:** To predict the likelihood of diabetes based on relevant health metrics.
*   **Input Data:** Tabular data containing various health-related features. 
*   **Output:** A binary prediction:
    *   `0` representing the absence of diabetes.
    *   `1` representing the presence of diabetes.
*   **Dataset:** The model is trained and evaluated on a custom dataset made from Kaggle's Diabetes prediction dataset ([https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset])
*   **Model Training:**
    *   The dataset is split into training and testing sets (80% train, 20% test).
    *   A Random Forest Classifier is used.
    *   **Hyperparameter Tuning:** `GridSearchCV` is employed to find the best hyperparameters for the Random Forest model from a specified parameter grid. The grid includes parameters like `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. Cross-validation (cv=5) is used during the grid search.
    *   The model is trained on the training data using the best parameters found by `GridSearchCV`.
*   **Evaluation:**
    *   The best performing model from `GridSearchCV` is evaluated on the held-out test set.
    *   **Confusion Matrix:** A confusion matrix is printed to show the counts of true positive, true negative, false positive, and false negative predictions.
    *   **Classification Report:** A classification report is printed, providing metrics such as precision, recall, F1-score, and support for both classes.
    *   **Feature Importance:** The script calculates, prints, and visualizes the importance of each feature based on the trained Random Forest model. This helps in understanding which features contributed most to the predictions.
*   **Model File:** The best performing model found by `GridSearchCV` is saved to a file (`best_diabetes_rf.pkl`) using `joblib`.
*   **Dependencies:** pandas, scikit-learn, matplotlib, joblib.
