# Customer-Churn-Prediction

## Project Description

The Customer Churn Prediction web app is an end-to-end machine learning solution that predicts customer churn. The project involves building a complete pipeline, from data preprocessing and model training to hyperparameter tuning and deploying a web app for model inference. The goal is to help businesses identify customers who are likely to churn and take proactive measures to retain them.

## Features

- **Data Loading and Cleaning:** Load the customer dataset, handle missing values, and perform data preprocessing.
- **Model Training:** Train five different machine learning models (e.g., Gradient Boost, K-Nearest Neighbors, Random Forest, XGBoost, and Support Vector Machine) to predict customer churn.
- **Hyperparameter Tuning:** Optimize model performance using grid search or random search techniques to find the best parameters.
- **Model Serving:** Deploy the best-performing model through a web app, allowing users to make real-time predictions on customer churn probability.

## Tech Stack Used

- **Python:** Core programming language for the project.
- **Pandas:** Library for data manipulation and analysis.
- **Matplotlib:** Visualization library for plotting graphs and understanding data distribution.
- **scikit-learn:** Library for machine learning algorithms and model evaluation.
- **Streamlit:** Framework for building and deploying the web app.
- **groq:** Query language used to interact with the LLM for explanations and insights.

## How to Run the Project

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run app.py`.
4. Access the web app at `http://localhost:8501` in your web browser.

## Project Structure

- `app.py`: Streamlit app script to serving the model.
- `data/`: Directory containing the dataset.
- `models/`: Directory where the jupyter notebook was used for model development and saved trained models.
- `.streamlit/`: Directory containing the theme configuration.
- `requirements.txt`: List of dependencies.
