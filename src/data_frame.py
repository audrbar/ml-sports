import pandas as pd

# Load the CSV file into a DataFrame
filename = "sports_illustrated_articles.csv"
try:
    df = pd.read_csv(filename)
    print(f"File '{filename}' successfully loaded.")
except FileNotFoundError:
    print(f"File '{filename}' not found. Please check the path.")
    exit()

# Display initial rows
print("Initial DataFrame:")
print(df.head())

# Step 1: Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Step 2: Handle missing values
# Drop rows with missing values or fill them with placeholders
df_cleaned = df.dropna()  # Alternatively, use df.fillna("Unknown") to fill missing values
print("\nAfter handling missing values:")
print(df_cleaned.isnull().sum())

# Step 3: Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()
print(f"\nAfter removing duplicates: {len(df_cleaned)} records remaining.")

# Step 4: Clean text columns (e.g., strip extra whitespace)
text_columns = ['title', 'link', 'category']  # Adjust based on your DataFrame columns
for col in text_columns:
    df_cleaned[col] = df_cleaned[col].str.strip()

# Step 5: Verify link validity (optional)
# Ensure links start with "http" or "https" (basic validation)
df_cleaned = df_cleaned[df_cleaned['link'].str.startswith(('http', 'https'))]

# Step 6: Normalize categories (optional)
# Example: Convert categories to lowercase
df_cleaned['category'] = df_cleaned['category'].str.lower()

# Step 7: Reindex the DataFrame
df_cleaned.reset_index(drop=True, inplace=True)

# Display cleaned DataFrame summary
print("\nCleaned DataFrame Summary:")
print(df_cleaned.info())

# Display the first few rows of the cleaned DataFrame
print("\nCleaned DataFrame Sample:")
print(df_cleaned.head())

# Step 8: Save the cleaned DataFrame to a new CSV file
cleaned_filename = "cleaned_sports_illustrated_articles.csv"
df_cleaned.to_csv(cleaned_filename, index=False)
print(f"\nCleaned data saved to '{cleaned_filename}'")

# Adjust pandas settings to display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Adjust the display width for better readability

# Display all rows and columns of the cleaned DataFrame
print("\nCleaned DataFrame:")
print(df_cleaned)
