import time

import pandas as pd
import streamlit as st

import res
import utils as ut

# Load the dataset
df = pd.read_csv("data/churn.csv")

# Set the page configuration title
st.set_page_config(page_title="Customer Churn Prediction")

# Set the style of the line decoration
st.markdown("""
<style>
	[data-testid="stDecoration"] {
		background-image: linear-gradient(90deg, #52ff76, #00b4d8);
	}
</style>""",
unsafe_allow_html=True)

# Set the title of the page
st.title("Customer Churn Prediction")

# Create a list of customers
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

# Create a selectbox to select a customer
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    # Get the selected customer ID and name
    selected_customer_id, selected_name = (
        int(selected_customer_option.split(" - ")[0]),
        selected_customer_option.split(" - ")[1],
    )

    # Get the selected customer details
    selected_customer = df[df["CustomerId"] == selected_customer_id].iloc[0]

    # Display two columns for the input fields and customer metrics
    col1, col2 = st.columns(2)

    # Create an input field dictionary
    input_field = {
        "CreditScore": 0,
        "Location": "",
        "Gender": "",
        "Age": 0,
        "Tenure": 0,
        "Balance": 0.0,
        "NumOfProducts": 0,
        "HasCrCard": False,
        "IsActiveMember": False,
        "EstimatedSalary": 0.0,
    }

    # Create the input fields for the customer details (Credit Score, Location, Gender, Age, Tenure) in the left column
    with col1:
        input_field["CreditScore"] = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer["CreditScore"]),
        )

        input_field["Location"] = st.selectbox(
            "Location",
            ["France", "Spain", "Germany"],
            index=["France", "Spain", "Germany"].index(selected_customer["Geography"]),
        )

        input_field["Gender"] = st.radio(
            "Gender",
            ["Male", "Female"],
            index=0 if selected_customer["Gender"] == "Male" else 1,
        )

        input_field["Age"] = st.number_input(
            "Age", min_value=18, max_value=100, value=int(selected_customer["Age"])
        )

        input_field["Tenure"] = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"]),
        )

    # Create the input fields for the customer details (Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary) in the right column
    with col2:
        input_field["Balance"] = st.number_input(
            "Balance", min_value=0.0, value=float(selected_customer["Balance"])
        )

        input_field["NumOfProducts"] = st.number_input(
            "Number of Products",
            min_value=0,
            max_value=10,
            value=int(selected_customer["NumOfProducts"]),
        )

        input_field["HasCrCard"] = st.checkbox(
            "Has Credit Card", value=bool(selected_customer["HasCrCard"])
        )

        input_field["IsActiveMember"] = st.checkbox(
            "Is Active Member", value=bool(selected_customer["IsActiveMember"])
        )

        input_field["EstimatedSalary"] = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"]),
        )

    # Create a button to predict the customer churn
    predict_button = st.button("Predict Customer Churn")

    # Check if the predict button is clicked
    if predict_button:
        # Display the selected customer details
        input_df, input_dict = res.prepare_input(input_field)

        probabilities, avg_probability, estimated_clv = res.make_prediction(input_df)

        # Display the prediction results
        st.markdown("---")
        st.subheader("Prediction Results")

        # Load the spinner while generating the prediction results
        with st.spinner("Generating prediction results..."):
            time.sleep(2)
            col1, col2 = st.columns(2)

            # Display the gauge chart in the left column
            with col1:
                fig = ut.create_gauge_chart(avg_probability)
                st.plotly_chart(fig, use_container_width=True)
                st.write(
                    f"The customer has a {avg_probability:.2%} chance of churning."
                )
                st.write(f"Estimated Customer Lifetime Value: ${estimated_clv:.2f}")

            # Display the model probability chart in the right column
            with col2:
                fig = ut.create_model_probability_chart(probabilities)
                st.plotly_chart(fig, use_container_width=True)

            fig = ut.create_customer_percentiles_chart(selected_customer_id)
            st.plotly_chart(fig, use_container_width=True)

        # Display the explanation of the prediction
        st.markdown("---")
        st.subheader("Explanation of Prediction")

        # Load the spinner while generating the explanation
        with st.spinner("Generating explanation..."):
            time.sleep(2)
            explanation = res.explain_prediction(
                avg_probability, input_dict, selected_name
            )

            st.write(explanation)

        # Display the personalized email
        st.markdown("---")
        st.subheader("Personalized Email")

        # Load the spinner while generating the email
        with st.spinner("Generating email..."):
            time.sleep(2)
            email_content = res.generate_email(
                avg_probability, input_dict, explanation, selected_name
            )

            st.write(email_content)
