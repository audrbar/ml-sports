import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import adjust_pandas_display, load_csv


def vectorize_with_tfidf(data_frame, text_column, max_features=5000):
    """Tokenizes and vectorizes the text using TF-IDF, replacing the 'text' column with vectorized data.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame containing the text column.
        text_column (str): The name of the column with text to vectorize.
        max_features (int): Maximum number of features for the TF-IDF matrix.

    Returns:
        pd.DataFrame: The original DataFrame with the 'text' column replaced by vectorized TF-IDF data.
    """
    # Initialize TF-IDF Vectorizer
    print("Vectorizing started...\n- TF-IDF vectorizer initialized;")
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform the text column
    print(f"- vectorizing {text_column} column;")
    tfidf_matrix = vectorizer.fit_transform(data_frame[text_column])

    # Replace the 'text' column with the TF-IDF matrix
    print(f"- replacing {text_column} column with vectors;")
    data_frame['text'] = list(tfidf_matrix.toarray())
    print("Vectorizing done.")

    return data_frame


# Run vectorizing
if __name__ == "__main__":
    try:
        # Adjust pandas display settings
        adjust_pandas_display(max_rows=None, max_columns=None, width=1000)

        # Load and preprocess dataset
        df = load_csv("articles_preprocessed.csv")

        # Vectorize the text column
        vectorized_df = vectorize_with_tfidf(df, text_column="text")

        # Display the vectorized DataFrame
        print(f"\nVectorized DataFrame:\n{vectorized_df.head()}")

    except Exception as e:
        print(f"An error occurred during vectorizing: {e}")
