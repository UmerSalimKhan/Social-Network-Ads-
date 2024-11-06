# Social Network Ads Prediction

![Ad Campaign](https://cdn4.vectorstock.com/i/1000x1000/90/63/ad-campaign-vector-39019063.jpg)

This project aims to predict the likelihood of a user purchasing a product based on their information from a social network advertisement. It uses various machine learning models and techniques to train and evaluate the best-performing model. The final model is used for real-time prediction via a Streamlit web interface.

## Table of Contents

1.  [Dataset Used](#dataset-used
2.  [Models Used](#models-used)
3.  [Final Selected Model](#final-selected-model)
4.  [Libraries Used](#libraries-used)
5.  [Data Preprocessing](#data-preprocessing)
6.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7.  [Front-End Implementation](#front-end-implementation)
8.  [Future Work](#future-work)

## Dataset Used: 
- https://www.kaggle.com/datasets/rakeshrau/social-network-ads

## Models Used:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Gaussian Naive Bayes**

## Final Selected Model:
- **K-Nearest Neighbors (KNN)** was selected as the final model after testing multiple models with **K-Fold Cross-Validation (10 folds)**. KNN provided the best **mean accuracy** with minimal fluctuation in performance across folds.
- **Hyperparameter Tuning**: The accuracy was further improved using **Randomized Search CV**, resulting in an accuracy increase from around **91% to 95%**.

## Libraries Used:
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-Learn**

## Data Preprocessing:
- **Standard Scaling** was applied to the features to standardize the data before training the models.
  
## Exploratory Data Analysis (EDA):
- Data Exploration: Understanding the dataset and checking for missing values.
- Data Manipulation: Converting categorical data into numerical representations (e.g., Gender column).
- Data Cleaning: Handling missing data and duplicates. Luckily there was no missing data & duplicates.
- Data Visualization: Visualizing the distributions and correlations of features.

## Front-End Implementation:
- The front-end of the project is built using **Streamlit** in Google Colab, enabling easy interaction with the trained model for predictions.
- The **Joblib** library is used to dump and load the trained KNN model for future use in the web application.

## Future Work:
1. **Remove the Gender column**: The Gender column has a very low correlation with the target variable, which might not contribute much to the model's accuracy.
2. **Train model without Gender column**: After removing the column, retrain the model without the Gender feature.
3. **Evaluate the model**: Re-evaluate the performance of the model without the Gender feature and compare the results.
