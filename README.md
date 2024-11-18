Customer Churn Prediction Web App

Overview

This project is a Customer Churn Prediction Web Application built using Streamlit. The app utilizes multiple machine learning models to predict the probability of customer churn and provides actionable insights for bank managers to retain customers. It also generates personalized emails to engage customers based on their risk levels.

Features

Customer Data Input:
Editable input fields for customer demographic and financial details.
Pre-populated with selected customer's information for convenience.
Churn Prediction:
Predicts the likelihood of a customer churning using:
XGBoost
Random Forest
K-Nearest Neighbors
Displays predictions visually using:
Gauge Chart (average probability of churn).
Bar Chart (individual model probabilities).
Insights and Explanations:
Explains the prediction based on key features and customer data.
Utilizes the OpenAI API for generating clear and concise explanations.
Email Generation:
Creates a personalized email to customers to address churn risk.
Lists tailored incentives based on customer information.
Data Analysis:
Provides summary statistics for churned vs. non-churned customers.

Customer-Churn-Prediction/
│
├── data/
│   ├── churn.csv                # Dataset containing customer information
│
├── models/
│   ├── xgb_model.pkl            # Pre-trained XGBoost model
│   ├── nb_model.pkl             # Pre-trained Naive Bayes model
│   ├── rf_model.pkl             # Pre-trained Random Forest model
│   ├── dt_model.pkl             # Pre-trained Decision Tree model
│   ├── svm_model.pkl            # Pre-trained SVM model
│   ├── knn_model.pkl            # Pre-trained KNN model
│   ├── xgboost-SMOTE.pkl        # Pre-trained XGBoost with SMOTE
│   ├── xgboost-featureEngineered.pkl  # Pre-trained XGBoost with feature engineering
│
├── utils.py                     # Helper functions for creating visualizations
│
├── app.py                       # Main Streamlit application script
│
└── README.md                    # Documentation

Requirements

Software and Libraries
Python 3.8 or higher
Streamlit
Pandas
NumPy
OpenAI Python SDK
Plotly
Scikit-learn
Pickle

Usage

Select a Customer:
Use the dropdown menu to choose a customer from the dataset.
Modify Inputs:
Adjust customer details such as age, balance, or location to explore different scenarios.
View Predictions:
Check the churn probability and visualization for the selected customer.
Explanation:
Read a detailed explanation of the churn prediction.
Email Generation:
Review a personalized email generated for the customer with incentives to stay.
Key Functions

prepare_input()
Prepares user input into a DataFrame suitable for machine learning models.

make_predictions()
Uses multiple pre-trained models to predict churn probabilities and visualizes results.

explain_prediction()
Generates a natural language explanation of the prediction using OpenAI's API.

generate_email()
Creates a personalized email to the customer with loyalty incentives.

Visualizations

Gauge Chart
Displays the average churn probability.

Model Probability Chart
Compares individual model predictions for churn probability.

Customization

Model Integration:
Add or replace models in the models/ directory and modify the make_predictions() function.
Styling:
Enhance the UI by modifying the st.title, st.markdown, and other Streamlit components.
Dataset:
Replace churn.csv with your own dataset. Ensure the format matches the required schema.
Future Enhancements

Expand Features:
Add more features for a deeper analysis of customer behavior.
Enhanced Explanations:
Provide richer visual explanations of feature importance.
Email Integration:
Connect the app to an email-sending service for direct customer engagement.
License

This project is licensed under the MIT License.

Acknowledgments

OpenAI for their powerful API.
Streamlit for an easy-to-use interface framework.
The creators of the pre-trained machine learning models.
