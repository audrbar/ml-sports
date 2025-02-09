import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from utils import adjust_pandas_display, DATA_DIR

# Define paths for storing data
DATA_FILES = {
    "TF-IDF": os.path.join(DATA_DIR, "articles_vectorized_tfidf.pkl"),
    "Word2Vec": os.path.join(DATA_DIR, "articles_vectorized_word2vec.pkl"),
    "N-Gram": os.path.join(DATA_DIR, "articles_vectorized_ngram.pkl"),
    "FastText": os.path.join(DATA_DIR, "articles_vectorized_fasttext.pkl"),
}


def load_data(file_path):
    """Loads vectorized text features and labels from a pickle file."""
    try:
        df = pd.read_pickle(file_path).dropna()
        if not {"text", "category"}.issubset(df.columns):
            raise ValueError("Missing required columns: 'text' and 'category'")

        X = np.array(df["text"].tolist(), dtype=np.float32)
        y = LabelEncoder().fit_transform(df["category"])
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Splits the dataset into training, validation, and test sets."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def search_params(X_train, y_train):
    """Fine-tunes KNeighborsClassifier using Grid Search with cross-validation."""
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(
        KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search


def cross_validate_model(model, X_train, y_train, cv_folds=5):
    """Performs cross-validation on the given model."""
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"Mean Cross-Validation Accuracy: {np.mean(scores):.4f}")
    return scores


def evaluate_model(model, X, y, dataset_name="Test", show_matrix=True):
    """Evaluates model and prints key metrics."""
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)

    print(f"\n{dataset_name} Set Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if show_matrix:
        plot_confusion_matrix(confusion_matrix(y, y_pred), dataset_name)

    return [accuracy, precision, recall, f1]


def plot_confusion_matrix(conf_matrix, dataset_name):
    """Plots the confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {dataset_name} Set")
    plt.show()


def plot_metrics_comparison(results):
    """Plots comparison of test metrics across datasets."""
    df_results = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"])
    df_results.plot(kind='bar', figsize=(12, 8))
    plt.title("Model Performance Across Different Vectorization Methods")
    plt.ylabel("Score")
    plt.legend(title="Dataset")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    adjust_pandas_display(max_rows=None, max_columns=None)

    results = {}

    for dataset_key, file_path in DATA_FILES.items():
        print(f"\nTraining model on {dataset_key} dataset...")
        X, y = load_data(file_path)
        if X is None or y is None:
            continue

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        grid_search = search_params(X_train, y_train)
        cross_validate_model(grid_search.best_estimator_, X_train, y_train)
        test_scores = evaluate_model(grid_search.best_estimator_, X_test, y_test, dataset_name=dataset_key)
        results[dataset_key] = test_scores

    print("\nFinal Model Performance Comparison:")
    print(pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"]))
    plot_metrics_comparison(results)
