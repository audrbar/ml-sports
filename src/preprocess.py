import re
import os
import string
import nltk
import logging
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from utils import adjust_pandas_display, load_csv, write_csv, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download stopwords if not already present
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define paths for storing data
ARTICLES_CLEANED_CSV = os.path.join(DATA_DIR, "articles_cleaned.csv")
ARTICLES_PREPROCESSED_CSV = os.path.join(DATA_DIR, "articles_preprocessed.csv")

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def normalize_text(text, lemmatize=True, stem=False):
    """Cleans input text by removing HTML tags, punctuation, stopwords, and extra whitespace."""
    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.replace("â", "")
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif stem:
        tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


def preprocess_dataset(data_frame, text_columns, target_column, output_file):
    """Processes DataFrame: combines text columns, normalizes text, and filters relevant columns."""
    logging.info(f"Preprocessing dataset: Combining {text_columns} into 'text' column.")
    data_frame['text'] = data_frame[text_columns].fillna("").agg(" ".join, axis=1)

    logging.info("Normalizing 'text' and target columns.")
    data_frame['text'] = data_frame['text'].apply(normalize_text)
    data_frame[target_column] = data_frame[target_column].apply(normalize_text)

    data_frame = data_frame[['text', target_column]]

    logging.info("Saving preprocessed dataset.")
    write_csv(data_frame, output_file)

    return data_frame


if __name__ == "__main__":
    try:
        adjust_pandas_display(max_rows=None, max_columns=None, width=1000)
        df = load_csv(ARTICLES_CLEANED_CSV)
        preprocessed_df = preprocess_dataset(df, ['title', 'content'],
                                             'category', ARTICLES_PREPROCESSED_CSV)
        logging.info(f"Preprocessed dataset shape: {preprocessed_df.shape}")
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")

# import re
# import os
# import string
# import nltk
# from bs4 import BeautifulSoup
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from utils import adjust_pandas_display, load_csv, write_csv, DATA_DIR
#
# # Download stopwords if not already present
# # nltk.download('stopwords')
# # nltk.download('punkt')
# # nltk.download('wordnet')
#
# # Define paths for storing data
# ARTICLES_CLEANED_CSV = os.path.join(DATA_DIR, "articles_cleaned.csv")
# ARTICLES_PREPROCESSED_CSV = os.path.join(DATA_DIR, "articles_preprocessed.csv")
#
# # Initialize lemmatizer and stemmer
# lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
#
#
# def normalize_text(text, lemmatize=True, stem=False):
#     """Cleans the input text by removing HTML tags, punctuation, stopwords, and extra whitespace.
#
#     Parameters:
#         text (str): The text to be cleaned.
#         lemmatize (bool): Whether to apply lemmatization (default: True).
#         stem (bool): Whether to apply stemming (default: False).
#     Returns:
#         str: The cleaned text.
#     """
#     if not isinstance(text, str):
#         return ""  # Return an empty string if the input is not a valid string
#
#     # Remove HTML tags
#     text = BeautifulSoup(text, "html.parser").get_text()
#
#     # Remove specific unwanted symbols
#     text = text.replace("â", "")
#
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#
#     # Convert text to lowercase
#     text = text.lower()
#
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
#
#     # # Remove non-UTF-8 characters
#     # df['text'] = df['text'].apply(lambda x: x.encode('ascii', 'ignore').decode('utf-8'))
#
#     # Tokenize the text
#     tokens = word_tokenize(text)
#
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#
#     # Normalize tokens: lemmatization or stemming
#     if lemmatize:
#         tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     elif stem:
#         tokens = [stemmer.stem(word) for word in tokens]
#
#     # Join tokens back into a single string
#     normalized_text = ' '.join(tokens)
#
#     return normalized_text
#
#
# # Preprocessing function for DataFrame
# def preprocess_dataset(data_frame, text_columns, target_column, output_file):
#     """Combines specified text columns into a single 'text' column, removes rows with empty target column values,
#     applies normalization to the combined text, and outputs only 'text' and target columns.
#
#     Parameters:
#         data_frame (pd.DataFrame): The input DataFrame containing text and target columns.
#         text_columns (list of str): List of columns to combine into a single text column.
#         target_column (str): The name of the target column.
#         output_file (str): File name to save the preprocessed dataset.
#
#     Returns:
#         pd.DataFrame: The preprocessed DataFrame containing only 'text' and target columns.
#     """
#     # Step 1: Combine text columns into a single 'text' column
#     print(f"\nData preprocessing started...\n- combining text columns: {text_columns} into a single 'text' column.")
#     data_frame['text'] = data_frame[text_columns].fillna("").agg(" ".join, axis=1)
#
#     # Step 2: Normalize the 'text' column
#     print("- normalizing the 'text' column;")
#     data_frame['text'] = data_frame['text'].apply(normalize_text)
#     data_frame[target_column] = data_frame[target_column].apply(normalize_text)
#
#     # Step 3: Keep only 'text' and target column
#     print(f"- dropping {text_columns} columns;")
#     data_frame = data_frame[['text', target_column]]
#
#     # Step 4: Save the preprocessed DataFrame to a file
#     print(f"Data preprocessing done.")
#     write_csv(data_frame, output_file)
#
#     return data_frame
#
#
# # Run preprocessing
# if __name__ == "__main__":
#     try:
#         # Adjust pandas display settings
#         adjust_pandas_display(max_rows=None, max_columns=None, width=1000)
#
#         # Load and preprocess dataset
#         df = load_csv(ARTICLES_CLEANED_CSV)
#         preprocessed_df = preprocess_dataset(
#             df, ['title', 'content'], 'category', ARTICLES_PREPROCESSED_CSV
#         )
#
#         # Display the preprocessed DataFrame
#         print(f"\nPreprocessed DataFrame {preprocessed_df.shape}:\n{preprocessed_df.head()}")
#
#     except Exception as e:
#         print(f"An error occurred during preprocessing: {e}")
