import os
import random
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Load the machine learning model from the pickle file (.pkl)
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


# Load the dataset
df = pd.read_csv("data/churn.csv")

# Load the LLM client via Groq API
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

# Load the machine learning model
svm_model = load_model("models/svm_model.pkl")
xgb_model = load_model("models/xgb_model.pkl")
knn_model = load_model("models/knn_model.pkl")
rf_model = load_model("models/rf_model.pkl")
gb_model = load_model("models/gb_model.pkl")


# Function to prepare the input data for the model
def prepare_input(input_field):
    input_dict = {
        "CreditScore": input_field["CreditScore"],
        "Age": input_field["Age"],
        "Tenure": input_field["Tenure"],
        "Balance": input_field["Balance"],
        "NumOfProducts": input_field["NumOfProducts"],
        "HasCrCard": int(input_field["HasCrCard"]),
        "IsActiveMember": int(input_field["IsActiveMember"]),
        "EstimatedSalary": input_field["EstimatedSalary"],
        "Geography_France": 1 if input_field["Location"] == "France" else 0,
        "Geography_Germany": 1 if input_field["Location"] == "Germany" else 0,
        "Geography_Spain": 1 if input_field["Location"] == "Spain" else 0,
        "Gender_Male": 1 if input_field["Gender"] == "Male" else 0,
        "Gender_Female": 1 if input_field["Gender"] == "Female" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


# Function to make predictions and calculate the average probability
def make_prediction(input_df):
    probabilities = {
        "Gradient Boosting": gb_model.predict_proba(input_df)[0][1],
        "Random Forest": rf_model.predict_proba(input_df)[0][1],
        "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1],
        "XGBoost": xgb_model.predict_proba(input_df)[0][1].item(),
        "Support Vector Machine": svm_model.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    estimated_clv = (
        input_df["Balance"].values[0] * input_df["EstimatedSalary"].values[0] / 100000
    )

    return probabilities, avg_probability, estimated_clv

# Function to generate the explanation of the prediction
def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data science at a bank where you specialize in interpreting and explaining predictions of machine learning models.
    
Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

Here is the customer's information:
{input_dict}

Here are the machine learning model's top 10 most important features for predicting churn:

            Feature | Importance
--------------------|---------------------
      NumOfProducts | 0.323888
     IsActiveMember | 0.164146
                Age | 0.109550
  Geography_Germany | 0.091373
            Balance | 0.052786
   Geography_France | 0.046463
      Gender_Female | 0.045283
    Geography_Spain | 0.036855
        CreditScore | 0.035005
    EstimatedSalary | 0.032655
          HasCrCard | 0.031940
             Tenure | 0.030054
        Gender_Male | 0.000000

{pd.set_option('display.max_columns', None)}

Here are summary statistic for churned customers:
{df[df["Exited"] == 1].describe()}

Here are summary statistic for non-churned customers:
{df[df["Exited"] == 0].describe()}

- If the customer has over a 40% risk of churning, generate a explanation of why they are at risk of churning.
- If the customer has less than a 40% risk of churning, generate a explanation of why they might not be at risk of churning.
- Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

DO NOT mention the probability of churning, or the machine learning model, or say anthing like "Based on the machine learning model's prediction and top 10 most important features", just expain the prediction.

MAKE SURE IT NOT LONGER THAN 100 WORDS.
    """

    raw_response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
    )

    return raw_response.choices[0].message.content

# Function to generate an email to the customer
def generate_email(probability, input_dict, explanation, surname):
    manager_names = [
        "Ethan Delgado",
        "Victoria Langston",
        "Miles Chen",
        "Olivia Martinez",
        "Daniel Okafor",
        "Charlotte Nguyen",
        "Benjamin Hawthorne",
        "Amelia Rosario",
        "Jonathan Choi",
        "Isabella Jackson"
    ]

    random_name = random.choice(manager_names)

    prompt = f"""
    Your name is {random_name} and you are a manager at HS Bank, You are responsible for esuring customers start with the bank and are incentivized with various offers.
    
    You noticed a custome name {surname} has a {round(probability * 100, 1)}% probability of churning.

    Here is the customer's information:
    {input_dict}

    Here is some explanation as to why the customer might be at risk of churning: {explanation}

    Generate an email to the customer based on their inromation, asking them to stay if there are at risk of churning or offering them incentives so that they become more loyal to the bank.

    Make sure to list out a set of incentives to saty based on their information, in bullet point format, DO NOT ever mention the probability of churning, or the machine learning model to the customer.
    """

    raw_response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
    )

    return raw_response.choices[0].message.content
