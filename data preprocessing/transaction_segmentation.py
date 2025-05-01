import os
import pandas as pd

# Path to the transaction.csv file
input_file = '/Users/dipeshacharya/Desktop/hackathon/gbime_fraud_detection/data preprocessing/global_ime_bank_transactions.csv'

# Output directory for segmented files
output_dir = '/Users/dipeshacharya/Desktop/hackathon/gbime_fraud_detection/data preprocessing/segmented_transactions'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the transaction data
try:
    data = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"File not found: {input_file}")
    exit()

# Ensure 'transaction_date' is in datetime format
# data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')

# # Print the hour part of the 'transaction_date'
# data["hour"]=data["transaction_date"].dt.hour

# Ensure the file contains a 'customer_id' column
if 'customer_id' not in data.columns:
    print("The input file must contain a 'customer_id' column.")
    exit()

# Group the data by customer_id and save each user's transactions to a separate file
for customer_id, user_data in data.groupby('customer_id'):
    user_file = os.path.join(output_dir, f'user_{customer_id}_transactions.csv')
    user_data.to_csv(user_file, index=False)
    print(f"Saved transactions for user {customer_id} to {user_file}")

print("Segmentation complete.")

# Filter out transactions where 'is_suspicious' is False
if 'is_suspicious' not in data.columns:
    print("The input file must contain an 'is_suspicious' column.")
    exit()

filtered_data = data[data['is_suspicious'] == False]

# Save the filtered data to a single file
output_file = os.path.join(output_dir, 'non_suspicious_transactions.csv')
filtered_data.to_csv(output_file, index=False)
print(f"Saved non-suspicious transactions to {output_file}")

print("Non-suspicious segmentation complete.")
