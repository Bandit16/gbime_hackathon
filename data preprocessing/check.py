import pandas as pd

# Load the transaction CSV file
file_path = 'global_ime_bank_transactions.csv'
transactions = pd.read_csv(file_path)

# Check if necessary columns exist
required_columns = {'transaction_type', 'is_suspicious'}
if not required_columns.issubset(transactions.columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

summary = transactions.groupby('transaction_type').agg(
    total_count=('is_suspicious', 'count'),
    suspicious_count=('is_suspicious', lambda x: (x == True).sum())
)
summary['suspicious_percentage'] = (summary['suspicious_count'] / summary['total_count']) * 100

# Display the results
print("Suspicious Transaction Percentage by Transaction Type:")
print(summary)