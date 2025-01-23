import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from data_frame import adjust_pandas_display, load_csv
# Download stopwords if not already present
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def clean_text(text):
    """Cleans the input text by removing HTML tags, punctuation, stopwords, and extra whitespace.

    Parameters:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""  # Return an empty string if the input is not a valid string

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)

    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


# Preprocessing function for DataFrame
def preprocess_dataset(data_frame, text_columns, output_file="articles_preprocessed.csv"):
    """Preprocesses specified text columns in a DataFrame and saves the preprocessed dataset to a file.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame containing text columns.
        text_columns (list of str): List of columns to clean and preprocess.
        output_file (str): File name to save the preprocessed dataset.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Apply the cleaning function to each specified column
    for col in text_columns:
        if col in data_frame.columns:
            print(f"Cleaning column: {col}")
            data_frame[col] = data_frame[col].apply(clean_text)

    # Save the preprocessed DataFrame to a file
    data_frame.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")
    return data_frame


# Run preprocessing
if __name__ == "__main__":
    # Adjust pandas display settings
    adjust_pandas_display(max_rows=None, max_columns=None, width=1000)

    # Load a CSV file into a DataFrame
    df = load_csv("articles_cleaned.csv")

    # Preprocess the dataset
    preprocessed_df = preprocess_dataset(
        df, text_columns=['title', 'intro', 'author', 'category'], output_file="article_preprocessed.csv"
    )

    # Display the preprocessed DataFrame
    print(f"\nPreprocessed DataFrame:\n{preprocessed_df}")
