import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import adjust_pandas_display, check_class_distribution, load_csv, split_data, load_pkl


def train_and_evaluate_with_validation(data, X_train, X_val, X_test, y_train, y_val, y_test, model_type):
    """Builds and evaluates a classification model with multiple text columns.

    Parameters:
        data (pd.DataFrame): The dataset containing multiple text columns and labels.
        text_column (str): The column name containing text.
        label_column (str): The column name containing labels.
        model_type (str): The type of model to use ('logistic_regression', 'random_forest').

    Returns:
        None
    """
    # Step 1: Initialize the classifier
    if model_type == "logistic_regression":
        model = LogisticRegression(random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'logistic_regression' or 'random_forest'.")

    # Step 2: Train the model using the training set
    model.fit(X_train, y_train)

    # Step 3: Evaluate the model on the validation set
    y_test_predicted = model.predict(X_test)
    y_val_predicted = model.predict(X_val)

    print(f"\n=== Testing Set Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_test_predicted):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_predicted, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_predicted, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_test_predicted, average='weighted'):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_predicted, zero_division=0)}")

    # Step 4: Evaluate the final model on the test set
    print(f"\n=== Validation Set Evaluation ({model_type.replace('_', ' ').title()}) ===")
    print(f"Accuracy: {accuracy_score(y_val, y_val_predicted):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_predicted, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, y_val_predicted, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_val_predicted, average='weighted'):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_val, y_val_predicted, zero_division=0)}")


# Run classification
if __name__ == "__main__":
    try:
        # Adjust pandas display settings
        adjust_pandas_display(max_rows=None, max_columns=None)

        # Load and preprocess dataset
        df = load_pkl("articles_vectorized_tfidf.pkl").dropna()
        print(f"\nLoaded DataFrame:\n{df.head()}")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, "text", "category")

        # Train and evaluate models
        train_and_evaluate_with_validation(
            df, X_train, X_val, X_test, y_train, y_val, y_test, "logistic_regression"
        )
        train_and_evaluate_with_validation(
            df, X_train, X_val, X_test, y_train, y_val, y_test, "random_forest"
        )

    except Exception as e:
        print(f"An error occurred during classification: {e}")
