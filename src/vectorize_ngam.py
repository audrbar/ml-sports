import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import adjust_pandas_display, load_csv, write_pkl


def vectorize_ngrams(data_frame, text_column, ngram_range=(1, 2), max_features=5000):
    """Tokenizes and vectorizes the text using n-grams and TF-IDF.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame containing the text column.
        text_column (str): The name of the column with text to vectorize.
        ngram_range (tuple): The range of n-grams to include (e.g., (1, 2) for unigrams and bigrams).
        max_features (int): The maximum number of features for the TF-IDF matrix.

    Returns:
        pd.DataFrame: A DataFrame with n-gram-based TF-IDF vectorized representation in the 'text' column.
    """
    # Initialize TF-IDF Vectorizer with n-gram range
    print("Vectorizing started...\n- TF-IDF vectorizer initialized;")
    vectorizer_tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

    # Fit and transform the text column
    print(f"- vectorizing {text_column} column;")
    tfidf_matrix = vectorizer_tfidf.fit_transform(data_frame[text_column])

    # Replace the 'text' column with the vectorized n-gram representation
    print(f"- replacing {text_column} column with vectors;")
    data_frame[text_column] = list(tfidf_matrix.toarray())
    print("Vectorization done.")

    return data_frame, vectorizer_tfidf


# Run Word2Vec vectorizer
if __name__ == "__main__":
    try:
        # Adjust pandas display settings
        adjust_pandas_display(max_rows=None, max_columns=None)

        # Load and preprocess dataset
        df = load_csv("articles_preprocessed.csv")

        # Vectorize the text column using n-grams
        vectorized_df, vectorizer = vectorize_ngrams(
            data_frame=df,
            text_column="text",
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=1000
        )

        # Write vectorized DataFrame to file
        write_pkl(vectorized_df, "articles_vectorized_ngram")

        # Display the updated DataFrame
        print(f"\nDataFrame with N-Gram Vectors in 'text' Column:\n{vectorized_df.head()}")

        # Display feature names (n-grams)
        print(f"\nN-Gram Features:\n{vectorizer.get_feature_names_out()[:30]}")

    except Exception as e:
        print(f"An error occurred during vectorizing: {e}")
