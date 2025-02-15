import re
import os
import string
import nltk
import logging
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

# Define character replacements
CHAR_REPLACEMENTS = {
    "â": "",
    "s": "",
    "t": "",
    " ": "",
    "ve": "",
    " ": "",
    "â": "",
    "â ": "",
    "": " ",
    "": " ",
    "": "",
    "m": "",
    "re": "",
    "â©": "",
    "i": "",
}


def clean_special_characters(text):
    """Replaces unwanted character sequences with proper equivalents."""
    for old, new in CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def normalize_text(text, lemmatize=True, stem=False):
    """Cleans input text by removing HTML tags, punctuation, stopwords, and extra whitespace."""
    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = clean_special_characters(text)
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
