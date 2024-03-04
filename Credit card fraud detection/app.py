import streamlit as st
import pickle
import pandas as pd


model = pickle.load(open('credit_card_fraud_detection.pkl', 'rb'))
def main():
    st.title('Credit Card Fraud Detection')
    cc_num = st.text_input('Enter Credit Card Number:')
    merchant = st.text_input('Enter Merchant ID:')

    if st.button('Detect Fraud'):
        if not cc_num or not merchant:
            st.error('Please enter both credit card number and merchant ID.')
        else:
            # Preprocess input parameters and make prediction
            predicted_class = is_fraud (cc_num, merchant)
            if predicted_class == 1:
                st.error('Fraudulent transaction detected!')
            else:
                st.success('Transaction is not fraudulent.')

def is_fraud(credit_card_number, merchant_id):
    # Preprocess input parameters (if necessary)
    # For example, you may need to encode categorical features like merchant ID
    
    # Create a DataFrame with input parameters
    data = pd.DataFrame({'credit_card_number': [credit_card_number], 'merchant_id': [merchant_id]})
    
    # Make predictions using the loaded model
    predicted_class = model.predict(data)
    
    return predicted_class[0]  # Return the predicted class (fraudulent or non-fraudulent)

if __name__ == '__main__':
    main()                