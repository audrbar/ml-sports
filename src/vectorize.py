import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import adjust_pandas_display, load_csv


def vectorize_text_with_tfidf(data_frame, text_columns, output_file="articles_vectorized.csv", max_features=5000):
    """Tokenizes and vectorizes the text in specified columns using TF-IDF and saves the result to a file.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame containing text columns to vectorize.
        text_columns (list of str): List of column names to tokenize and vectorize.
        output_file (str): The name of the file to save the vectorized DataFrame.
        max_features (int): The maximum number of features for the TF-IDF matrix. Default is 5000.

    Returns:
        pd.DataFrame: The DataFrame containing the original data and vectorized text columns.
    """
    vectorized_dfs = []
    for col in text_columns:
        if col in data_frame.columns:
            print(f"Vectorizing column: {col}")
            # Initialize TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(max_features=max_features)

            # Fit and transform the text column
            tfidf_matrix = vectorizer.fit_transform(data_frame[col].fillna(""))

            # Create a DataFrame with the TF-IDF features
            feature_names = vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_{feat}" for feat in feature_names])

            vectorized_dfs.append(tfidf_df)
        else:
            print(f"Warning: Column '{col}' does not exist in the DataFrame.")

    # Concatenate all vectorized columns along with the original DataFrame
    tfidf_combined = pd.concat([data_frame.reset_index(drop=True)] + vectorized_dfs, axis=1)

    # Save the combined DataFrame to a file
    try:
        tfidf_combined.to_csv(output_file, index=False)
        print(f"Vectorized DataFrame saved successfully to '{output_file}'")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")

    return tfidf_combined


# Run vectorizing
if __name__ == "__main__":
    try:
        # Adjust pandas display settings
        adjust_pandas_display(max_rows=None, max_columns=None, width=1000)

        # Load and preprocess dataset
        df = load_csv("articles_preprocessed.csv")
        df = df.drop(columns=["link"])
        vectorized_df = vectorize_text_with_tfidf(
            df, text_columns=['title', 'intro', 'author', 'category']
        )
        # Display the vectorized DataFrame
        print(f"\nVectorized DataFrame:\n{vectorized_df.head()}")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
