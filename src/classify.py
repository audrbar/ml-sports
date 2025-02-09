import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    """Trains and evaluates a classifier and returns performance metrics."""
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro', zero_division=0)
    recall = recall_score(y_test, y_predict, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_predict, average='macro', zero_division=0)

    return {
        'Classifier': model.__class__.__name__,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }


def evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test):
    """Evaluates a list of classifiers and returns their performance metrics."""
    results = [evaluate_classifier(clf, X_train, X_test, y_train, y_test) for clf in classifiers]
    return pd.DataFrame(results).set_index('Classifier')


def plot_metrics(metrics_df):
    """Plots performance metrics for classifiers."""
    plt.figure(figsize=(10, 6))
    for column in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        plt.plot(metrics_df.index, metrics_df[column], marker='o', linestyle='-', label=column)

    plt.title("Classifiers Performance Metrics")
    plt.xlabel("Classifier")
    plt.ylabel("Metric Value")
    plt.ylim(0.2, 1.0)
    plt.legend(title="Metrics")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metrics_all(metrics_df):
    """Plots performance metrics for classifiers."""
    metrics_df.plot(kind='bar', figsize=(12, 7))
    plt.title("Classifiers Performance Metrics")
    plt.ylabel("Score")
    plt.legend(title="Metrics")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    adjust_pandas_display(max_rows=None, max_columns=None)

    results = {}
    classifiers_to_evaluate = [
        LogisticRegression(solver='liblinear', max_iter=500),
        DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42),
        RandomForestClassifier(n_estimators=40, max_depth=20, bootstrap=True),
        KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance'),
    ]

    for dataset_key, file_path in DATA_FILES.items():
        print(f"Training models on {dataset_key} dataset...")
        X, y = load_data(file_path)
        if X is None or y is None:
            continue

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        metrics_df = evaluate_classifiers(classifiers_to_evaluate, X_train, X_test, y_train, y_test)
        results[dataset_key] = metrics_df

        # print(f"\nPerformance Metrics for {dataset_key}:")
        # print(metrics_df)
        plot_metrics(metrics_df)

    print("\nFinal Model Performance Comparison Across Datasets:")
    for dataset_key, metrics_df in results.items():
        print(f"\nDataset: {dataset_key}")
        print(metrics_df)
        plot_metrics_all(metrics_df)
