import os
import pandas as pd

# Path to the transaction.csv file
input_file = '/Users/dipeshacharya/Desktop/hackathon/gbime_fraud_detection/data preprocessing/global_ime_bank_transactions.csv'

df = pd.read_csv(input_file)
for customer_id in df['customer_id'].unique():
    # Filter the DataFrame for the current customer_id
    print(customer_id)
