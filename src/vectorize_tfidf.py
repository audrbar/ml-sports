import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import adjust_pandas_display, load_csv, write_pkl


def vectorize_tfidf(data_frame, text_columns, max_features=5000):
    """Tokenizes and vectorizes multiple text columns using TF-IDF, replacing their content with vectorized data.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame containing the text columns.
        text_columns (list of str): List of column names to vectorize.
        max_features (int): Maximum number of features for the TF-IDF matrix.

    Returns:
        pd.DataFrame: The original DataFrame with specified columns replaced by vectorized TF-IDF data.
    """
    # Initialize TF-IDF Vectorizer
    print("\nVectorizing started...\n- TF-IDF vectorizer initialized;")
    vectorizer_tfidf = TfidfVectorizer(max_features=max_features)

    for column in text_columns:
        # Check if column exists in the DataFrame
        if column not in data_frame.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping...")
            continue

        # Fit and transform the column
        print(f"- vectorizing column: '{column}';")
        tfidf_matrix = vectorizer_tfidf.fit_transform(data_frame[column].fillna(""))

        # Replace the column with its vectorized representation
        print(f"- replacing '{column}' column with vectors;")
        data_frame[column] = list(tfidf_matrix.toarray())

    print("Vectorization specified column completed.")
    return data_frame


# Run TF-IDF vectorizer
if __name__ == "__main__":
    try:
        # Adjust pandas display settings
        adjust_pandas_display(max_rows=None, max_columns=None)

        # Load and preprocess dataset
        df_loaded = load_csv("articles_preprocessed.csv")

        # Vectorize the text column
        df_tfidf = vectorize_tfidf(df_loaded, ["text"])

        # Write vectorized DataFrame to file
        write_pkl(df_tfidf, "articles_vectorized_tfidf")

        # Display the vectorized DataFrame
        print(f"\nTF-IDF vectorized DataFrame:\n{df_tfidf.head()}")

    except Exception as e:
        print(f"An error occurred during vectorizing: {e}")
