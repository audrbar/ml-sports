import os
import pickle

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from utils import adjust_pandas_display, DATA_DIR, MODELS_DIR

# Define paths for storing data
DATA_FILES = {
    "TF-IDF": os.path.join(DATA_DIR, "articles_vectorized_tfidf.pkl"),
    "Word2Vec": os.path.join(DATA_DIR, "articles_vectorized_word2vec.pkl"),
    "N-Gram": os.path.join(DATA_DIR, "articles_vectorized_ngram.pkl"),
    "FastText": os.path.join(DATA_DIR, "articles_vectorized_fasttext.pkl"),
}
LR_MODEL = os.path.join(MODELS_DIR, "lr_trained_model.pkl")
LR_ENCODER = os.path.join(MODELS_DIR, "lr_label_encoder.pkl")


def load_data(file_path):
    """Loads vectorized text features and labels from a pickle file."""
    try:
        df = pd.read_pickle(file_path).dropna()
        if not {"text", "category"}.issubset(df.columns):
            raise ValueError("Missing required columns: 'text' and 'category'")

        X = np.array(df["text"].tolist(), dtype=np.float32)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["category"])
        return X, y, label_encoder  # Return encoder too
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def search_params(X_train, y_train):
    """Fine-tunes LogisticRegression using Grid Search with cross-validation."""
    param_grid = {
        'C': [10, 20, 30, 50],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 200, 300, 600]
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, dataset_name="Test", show_matrix=True):
    """Evaluates model and prints key metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
    }

    print(f"\n{dataset_name} Set Evaluation")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    if show_matrix:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), dataset_name)

    return metrics


def save_model_and_encoder(model, label_encoder, model_path, encoder_path):
    """Saves the trained LogisticRegression model and label encoder."""
    # Save model using joblib
    joblib.dump(model, model_path)
    # Save label encoder using pickle
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Model saved to {model_path}, Label encoder saved to {encoder_path}")


def plot_confusion_matrix(conf_matrix, dataset_name):
    """Plots the confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Logistic Regression Model Confusion Matrix - {dataset_name} Set")
    plt.show()


def plot_metrics_comparison(results):
    """Plots comparison of test metrics across datasets."""
    df_results = pd.DataFrame(results)
    df_results.T.plot(kind='bar', figsize=(12, 7))
    plt.title("Logistic Regression Model Performance Across Different Vectorization Methods")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    adjust_pandas_display(max_rows=None, max_columns=None)
    results = {}

    for dataset_key, file_path in DATA_FILES.items():
        print(f"\nTraining model on {dataset_key} dataset...")
        X, y, label_encoder = load_data(file_path)  # Retrieve label_encoder
        if X is None or y is None:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        best_model = search_params(X_train, y_train)
        test_scores = evaluate_model(best_model, X_test, y_test, dataset_name=dataset_key)
        results[dataset_key] = test_scores

        # Save the best model and label encoder
        save_model_and_encoder(best_model, label_encoder, LR_MODEL, LR_ENCODER)

    print("\nFinal Logistic Regression Model Performance Comparison:")
    print(pd.DataFrame(results))
    plot_metrics_comparison(results)
