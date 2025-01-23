import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from utils import adjust_pandas_display, load_csv

# Download stopwords if not already present
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# Initialize lemmatizer and stemmer
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def normalize_text(text, lemmatize=True, stem=False):
    """Cleans the input text by removing HTML tags, punctuation, stopwords, and extra whitespace.

    Parameters:
        text (str): The text to be cleaned.
        lemmatize (bool): Whether to apply lemmatization (default: True).
        stem (bool): Whether to apply stemming (default: False).
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""  # Return an empty string if the input is not a valid string

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove specific unwanted symbols
    text = text.replace("â", "")

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

    # Normalize tokens: lemmatization or stemming
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif stem:
        tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a single string
    normalized_text = ' '.join(tokens)

    return normalized_text


# Preprocessing function for DataFrame
def preprocess_dataset(data_frame, text_columns, output_file="articles_preprocessed.csv"):
    """Cleans and normalizes the input text by removing HTML tags, punctuation, stopwords,
    and normalizing text via lemmatization or stemming.

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
            data_frame[col] = data_frame[col].fillna("").apply(normalize_text)

    # Save the preprocessed DataFrame to a file
    data_frame.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")
    return data_frame


# Run preprocessing
if __name__ == "__main__":
    try:
        # Adjust pandas display settings
        adjust_pandas_display(max_rows=None, max_columns=None, width=1000)

        # Load and preprocess dataset
        df = load_csv("articles_cleaned.csv")
        preprocessed_df = preprocess_dataset(
            df, text_columns=['title', 'intro', 'author', 'category']
        )

        # Display the preprocessed DataFrame
        print(f"\nPreprocessed DataFrame:\n{preprocessed_df.head()}")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
