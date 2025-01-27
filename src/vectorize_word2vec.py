import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from utils import adjust_pandas_display, load_csv, write_pkl


def vectorize_with_word2vec(data_frame, text_column, vector_size=100, min_count=1):
    """Tokenizes text and generates document vectors using Word2Vec embeddings, replacing the 'text' column with vectors.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame containing the text column.
        text_column (str): The name of the column with text to vectorize.
        vector_size (int): Dimensionality of the word vectors.
        min_count (int): Minimum frequency for a word to be included in the Word2Vec model.

    Returns:
        pd.DataFrame: The original DataFrame with the 'text' column replaced by Word2Vec vectors.
    """
    # Tokenize the text column
    print("Vectorizing started...\n- text tokenizing;")
    tokenized_text = data_frame[text_column].apply(word_tokenize)

    # Train a Word2Vec model
    print("- model Word2Vec initializing;")
    model = Word2Vec(sentences=tokenized_text, vector_size=vector_size, min_count=min_count, window=5)

    # Create document vectors by averaging word vectors
    def document_vector(tokens):
        # Filter out tokens that are not in the Word2Vec vocabulary
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

    # Replace the 'text' column with document vectors
    print(f"- tokens filtering;\n- replacing {text_column} column with vectors;")
    data_frame[text_column] = tokenized_text.apply(document_vector)
    print("Vectorization done.")

    return data_frame


# Run Word2Vec vectorizer
if __name__ == "__main__":
    try:
        # Adjust pandas display settings
        adjust_pandas_display(max_rows=None, max_columns=None)

        # Load and preprocess dataset
        df = load_csv("articles_preprocessed.csv")

        # Vectorize the text column
        df_word2vec = vectorize_with_word2vec(df, text_column="text")

        # Write vectorized DataFrame to file
        write_pkl(df_word2vec, "articles_vectorized_word2vec")

        # Display the vectorized DataFrame
        print(f"\nVectorized DataFrame:\n{df_word2vec.head()}")

    except Exception as e:
        print(f"An error occurred during vectorizing: {e}")
