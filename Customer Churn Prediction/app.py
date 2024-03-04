import streamlit as st
import pickle

# Load the pre-trained model
with open("customer_churn_prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Define the Streamlit app
def main():
    # Set title of the app
    st.title("Customer Churn Prediction App")

    # Add a brief description
    st.write("This app predicts whether a customer will churn or not.")

    # Add input fields for user to enter data
    credit_score = st.slider("Credit Score", min_value=300, max_value=999, value=608)
    age = st.slider("Age", min_value=18, max_value=100, value=41)
    tenure = st.slider("Tenure", min_value=0, max_value=10, value=1)
    balance = st.slider("Balance", min_value=0, max_value=2500000, value=83807)
    num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
    has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    has_active_member = st.selectbox("Has Active Member", ["Yes", "No"])
    # estimated_salary = st.slider("Estimated Salary", min_value=0, max_value=200000, value=101349)

    # Convert categorical features to numerical values
    has_credit_card = 1 if has_credit_card == "Yes" else 0
    has_active_member = 1 if has_active_member == "Yes" else 0

    # Predict churn
    if st.button("Predict Churn"):
        # Prepare input data
        input_data = [[credit_score, age, tenure, balance, num_of_products, has_credit_card, has_active_member]]
        
        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction result
        if prediction[0] == 1:
            st.error("Customer is predicted to churn.")
        else:
            st.success("Customer is predicted not to churn.")

if __name__ == "__main__":
    main()
