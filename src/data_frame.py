import os
import re
import pandas as pd


def adjust_pandas_display(max_rows=None, max_columns=None, width=1000):
    """
    Adjusts pandas display settings to control the number of rows, columns, and display width.

    Parameters:
        max_rows (int, optional): Maximum number of rows to display. Default is None (show all rows).
        max_columns (int, optional): Maximum number of columns to display. Default is None (show all columns).
        width (int): Width of the display in characters. Default is 1000.
    """
    pd.set_option('display.max_rows', max_rows)  # Show all rows or up to max_rows
    pd.set_option('display.max_columns', max_columns)  # Show all columns or up to max_columns
    pd.set_option('display.width', width)  # Adjust the display width
    print("Pandas display settings adjusted.")


def load_csv(filename):
    """
    Loads a CSV file into a DataFrame. Exits the program if the file is not found.

    Parameters:
        filename (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        data_frame = pd.read_csv(filename)
        print(f"File '{filename}' successfully loaded.")
        print(data_frame.info())
        print(data_frame.head())
        return data_frame
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please check the path.")
        exit()


def preprocess_dataframe(data_frame, text_columns):
    """
    Preprocesses a DataFrame by handling missing values, removing duplicates,
    cleaning text columns, and reindexing.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame to preprocess.
        text_columns (list of str): List of column names to clean (e.g., stripping whitespace).

    Returns:
        pd.DataFrame: The cleaned and reindexed DataFrame.
    """
    # Check for missing values
    print("\nChecking for missing values:")
    missing_values = data_frame.isnull().sum()
    print(missing_values)

    # Handle missing values only if they exist
    if missing_values.any():
        print("\nMissing values found. Dropping rows with missing values...")
        df_cleaned = data_frame.dropna()
    else:
        print("\nNo missing values found. Skipping dropna.")
        df_cleaned = data_frame.copy()

    # Remove duplicates
    initial_len = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"\nAfter removing duplicates: {len(df_cleaned)} records "
          f"remaining (removed {initial_len - len(df_cleaned)} duplicates).")

    # Clean text columns
    for col in text_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].str.strip()

    # Reindex the DataFrame
    df_cleaned.reset_index(drop=True, inplace=True)
    print("\nDataFrame reindexed.")

    # Display the cleaned DataFrame
    print("\nCleaned DataFrame:")
    print(df_cleaned)

    return df_cleaned


def add_category_column(data_frame, link_column='link', category_column='category', author_column='author'):
    """
    Extracts categories from a link column, cleans the author column, and updates the DataFrame.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame containing the link and author columns.
        link_column (str): The name of the column containing URLs. Default is 'link'.
        category_column (str): The name of the new column to store categories. Default is 'category'.
        author_column (str): The name of the column to clean (extract text before '|'). Default is 'author'.

    Returns:
        pd.DataFrame: The updated DataFrame with the new category column and cleaned author column.
    """

    # Define a function to extract category from a URL
    def extract_category(url):
        match = re.search(r"https://www\.si\.com/([^/]+)/", url)
        return match.group(1) if match else None

    # Apply the extraction function to the specified column
    data_frame[category_column] = data_frame[link_column].apply(extract_category)

    # Clean the author column to extract text before '|'
    if author_column in data_frame.columns:
        data_frame[author_column] = data_frame[author_column].str.split('|').str[0].str.strip()

    return data_frame


def cleaned_df_to_csv(data_frame, filename, append=True):
    """
    Saves a DataFrame to a CSV file. Appends to the file if it already exists.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the CSV file.
        append (bool): Whether to append to the file if it exists. Defaults to True.

    Returns:
        None
    """
    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    if file_exists and append:
        # Append to the existing file
        data_frame.to_csv(filename, mode='a', index=False, header=False)
        print(f"Data appended to '{filename}'.")
    else:
        # Overwrite or create a new file
        data_frame.to_csv(filename, index=False)
        print(f"Data saved to '{filename}'.")

    # Display the DataFrame saving result
    print(f"\nDataFrame saved.")


# Run the preprocessing
if __name__ == "__main__":
    # Adjust pandas display settings
    adjust_pandas_display(max_rows=None, max_columns=None, width=1000)

    # Load a CSV file into a DataFrame
    df = load_csv("articles_scraped.csv")

    # Preprocess the DataFrame
    df_clean = preprocess_dataframe(df, ['title', 'intro', 'author', 'link'])

    # Update DataFrame with the new category column
    df_updated = add_category_column(df_clean)

    # Save to CSV
    cleaned_df_to_csv(df_clean, "articles_cleaned.csv", append=True)
