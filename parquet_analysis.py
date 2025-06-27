import pandas as pd

# Path to your Parquet file
parquet_file_path = 'your_file.parquet'

# Load the Parquet file into a Pandas DataFrame
df = pd.read_parquet(parquet_file_path)

# Display the first few rows of the DataFrame
print(df.head())