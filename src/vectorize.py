import os
import pandas as pd
import numpy as np
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import adjust_pandas_display, load_csv, write_pkl, DATA_DIR

# Define paths for storing data
ARTICLES_PREPROCESSED_CSV = os.path.join(DATA_DIR, "articles_preprocessed.csv")
ARTICLES_VECTORIZED_TFIDF = os.path.join(DATA_DIR, "articles_vectorized_tfidf")
ARTICLES_VECTORIZED_FASTTEXT = os.path.join(DATA_DIR, "articles_vectorized_fasttext")
ARTICLES_VECTORIZED_NGRAM = os.path.join(DATA_DIR, "articles_vectorized_ngram")


def vectorize_tfidf(data_frame, text_columns, max_features=5000):
    """Vectorizes text columns using TF-IDF."""
    print("\nVectorizing started... TF-IDF vectorizer initialized;")
    vectorizer = TfidfVectorizer(max_features=max_features)

    for column in text_columns:
        if column in data_frame.columns:
            print(f"- Vectorizing column: '{column}';")
            tfidf_matrix = vectorizer.fit_transform(data_frame[column].fillna(""))
            data_frame[column] = list(tfidf_matrix.toarray())
        else:
            print(f"Warning: Column '{column}' not found. Skipping...")

    print("TF-IDF vectorization completed.")
    return data_frame


def vectorize_ngrams(data_frame, text_column, ngram_range=(1, 2), max_features=5000):
    """Vectorizes text using n-grams and TF-IDF."""
    print("\nVectorizing started... TF-IDF n-gram vectorizer initialized;")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(data_frame[text_column].fillna(""))
    data_frame[text_column] = list(tfidf_matrix.toarray())
    print("N-Gram vectorization completed.")
    return data_frame, vectorizer


def vectorize_with_fasttext(data_frame, text_column, vector_size=100, min_count=1, epochs=10):
    """Generates document embeddings using FastText."""
    print("\nVectorizing started... Training FastText;")
    tokenized_text = data_frame[text_column].apply(lambda x: word_tokenize(x.lower()) if isinstance(x, str) else [])

    # Train FastText model
    model = FastText(sentences=tokenized_text, vector_size=vector_size, min_count=min_count, window=5, sg=1,
                     epochs=epochs)

    def document_vector(tokens):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

    data_frame[text_column] = tokenized_text.apply(document_vector)
    print("FastText vectorization completed.")
    return data_frame


if __name__ == "__main__":
    try:
        adjust_pandas_display(max_rows=None, max_columns=None)
        df = load_csv(ARTICLES_PREPROCESSED_CSV)

        df_tfidf = vectorize_tfidf(df.copy(), ["text"])
        write_pkl(df_tfidf, ARTICLES_VECTORIZED_TFIDF)

        df_ngrams, _ = vectorize_ngrams(df.copy(), "text", ngram_range=(1, 2), max_features=1000)
        write_pkl(df_ngrams, ARTICLES_VECTORIZED_NGRAM)

        df_fasttext = vectorize_with_fasttext(df.copy(), text_column="text")
        write_pkl(df_fasttext, ARTICLES_VECTORIZED_FASTTEXT)

        print("\nAll vectorizations completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
