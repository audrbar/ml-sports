import re
import os
import pandas as pd
from utils import adjust_pandas_display, load_csv, write_csv, DATA_DIR

# Define paths for storing data
ARTICLES_SCRAPED_CSV = os.path.join(DATA_DIR, "articles_scraped.csv")
ARTICLES_CLEANED_CSV = os.path.join(DATA_DIR, "articles_cleaned.csv")


def handle_values(data_frame, text_columns):
    """Preprocesses a DataFrame by handling missing values, removing duplicates,
    cleaning text columns, and reindexing.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame to preprocess.
        text_columns (list of str): List of column names to clean (e.g., stripping whitespace).

    Returns:
        pd.DataFrame: The cleaned and reindexed DataFrame.
    """

    # Check for missing values
    print("\nHandling DataFrame values:")
    missing_values = data_frame.isnull().sum()

    # Handle missing values only if they exist
    if missing_values.any():
        print("- missing values found. Dropping rows with missing values...")
        df_cleaned = data_frame.dropna()
    else:
        print("- no missing values found. Skipping dropna.")
        df_cleaned = data_frame.copy()

    # Remove duplicates
    initial_len = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"- {len(df_cleaned)} records remaining after removing {initial_len - len(df_cleaned)} duplicates.")

    # Clean text columns
    for col in text_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].str.strip()

    # Reindex the DataFrame
    df_cleaned.reset_index(drop=True, inplace=True)
    print("- DataFrame reindexed.\nHandling DataFrame values done.")

    return df_cleaned


def category_column(data_frame, link_column='url', category_column_='category'):
    """Extracts categories from a link column, cleans the author column, and updates the DataFrame.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame containing the link and author columns.
        link_column (str): The name of the column containing URLs. Default is 'link'.
        category_column_ (str): The name of the new column to store categories. Default is 'category'.

    Returns:
        pd.DataFrame: The updated DataFrame with the new category column and cleaned author column,
        without the link column.
    """

    # Define a function to extract category from a URL
    def extract_category(url):
        match = re.search(r"https://www\.si\.com/([^/]+)/", url)
        return match.group(1) if match else None

    # Apply the extraction function to the link column
    data_frame[category_column_] = data_frame[link_column].apply(extract_category)
    # Drop the link column
    data_frame = data_frame.drop(columns=[link_column])

    return data_frame


def category_distribution(data_frame, target_column, min_count=5):
    """Checks the target class distribution and drops rows where the class count is less than a specified minimum.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame containing the target column.
        target_column (str): The name of the target column.
        min_count (int): The minimum count for a class to be retained. Default is 5.

    Returns:
        pd.DataFrame: A DataFrame with rare classes removed.
    """
    # Step 1: Calculate class distribution
    class_counts = data_frame[target_column].value_counts()
    print(f"\nClass Distribution before filtering:\n{class_counts}")

    # Step 2: Identify classes with count < min_count
    rare_classes = class_counts[class_counts < min_count].index.tolist()

    if rare_classes:
        print(f"\nClasses with fewer than {min_count} samples: {rare_classes}")
        # Step 3: Filter out rows with rare classes
        filtered_data = data_frame[~data_frame[target_column].isin(rare_classes)]
        print(f"\nClass Distribution after filtering:\n{filtered_data[target_column].value_counts()}")
    else:
        print(f"\nNo classes with fewer than {min_count} samples found.")
        filtered_data = data_frame

    initial_len = len(data_frame)
    print(f"\n{len(filtered_data)} records remaining after filtering {initial_len - len(filtered_data)} "
          f"rows with fewer than {min_count} samples in classes: {rare_classes}.")

    return filtered_data


# Run the values handling
if __name__ == "__main__":
    # Adjust pandas display settings
    adjust_pandas_display(max_rows=None, max_columns=None, width=1000)

    # Load a CSV file into a DataFrame
    df = load_csv(ARTICLES_SCRAPED_CSV)

    # Handle missing values, remove duplicates, clean text columns and reindex
    df_clean = handle_values(df, ['url', 'title', 'content'])

    # Update DataFrame with the new category column
    df_category = category_column(df_clean)

    # Filter the DataFrame based on class distribution
    df_distribution = category_distribution(df_category, target_column="category", min_count=8)

    # Save to CSV
    write_csv(df_distribution, ARTICLES_CLEANED_CSV)
