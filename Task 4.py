import pandas as pd

# Load datasets
csv_dataset = pd.read_csv('/content/sample_data/california_housing_train.csv')
json_dataset = pd.read_json('/content/sample_data/anscombe.json')

# Display the first 10 and last 10 rows of each dataset
print("CSV Dataset - First 10 rows:\n", csv_dataset.head(10))
print("CSV Dataset - Last 10 rows:\n", csv_dataset.tail(10))
print("JSON Dataset - First 10 rows:\n", json_dataset.head(10))
print("JSON Dataset - Last 10 rows:\n", json_dataset.tail(10))

# Display unique values and the number of unique values in a specific column
csv_column = 'your_specific_column'  # Replace with your actual column name
json_column = 'your_specific_column'  # Replace with your actual column name
print("CSV Dataset - Unique values in column:", csv_dataset[csv_column].unique())
print("CSV Dataset - Number of unique values in column:", csv_dataset[csv_column].nunique())
print("JSON Dataset - Unique values in column:", json_dataset[json_column].unique())
print("JSON Dataset - Number of unique values in column:", json_dataset[json_column].nunique())

# Generate descriptive statistics
print("CSV Dataset - Descriptive statistics:\n", csv_dataset.describe())
print("JSON Dataset - Descriptive statistics:\n", json_dataset.describe())

# Display concise summaries of each dataset
print("CSV Dataset - Summary info:\n", csv_dataset.info())
print("JSON Dataset - Summary info:\n", json_dataset.info())

# Identify columns with missing data
print("CSV Dataset - Columns with missing data:\n", csv_dataset.isnull().sum())
print("JSON Dataset - Columns with missing data:\n", json_dataset.isnull().sum())

# Fill missing values in numerical columns
csv_dataset[csv_column].fillna(csv_dataset[csv_column].mean(), inplace=True)
json_dataset[json_column].fillna(json_dataset[json_column].mean(), inplace=True)

# Fill missing values in categorical columns with the mode
csv_dataset['your_categorical_column'].fillna(csv_dataset['your_categorical_column'].mode()[0], inplace=True)
json_dataset['your_categorical_column'].fillna(json_dataset['your_categorical_column'].mode()[0], inplace=True)

# Drop rows and columns with missing values
csv_dataset_dropped = csv_dataset.dropna()
json_dataset_dropped = json_dataset.dropna()

# Manipulate DataFrames
csv_dataset['new_column'] = csv_dataset['existing_column1'] + csv_dataset['existing_column2']
json_dataset['new_column'] = json_dataset['existing_column1'] + json_dataset['existing_column2']

csv_dataset.drop(columns=['column_to_drop'], inplace=True)
json_dataset.drop(columns=['column_to_drop'], inplace=True)

csv_dataset.drop(index=[row_index_to_drop], inplace=True)
json_dataset.drop(index=[row_index_to_drop], inplace=True)

# Separate features and target variable
X = csv_dataset.drop(columns=['target_column'])
y = csv_dataset['target_column']

# Display the first 5 rows of X and y
print("Features (X) - First 5 rows:\n", X.head())
print("Target (y) - First 5 rows:\n", y.head())
