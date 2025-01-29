import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import adjust_pandas_display, load_pkl, split_data


def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    """Trains and evaluates a classifier and returns performance metrics.

    Parameters:
        model (sklearn model): The classifier to train and evaluate.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro', zero_division=0)
    recall = recall_score(y_test, y_predict, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_predict, average='macro', zero_division=0)

    return {
        'Classifier': model.__class__.__name__,
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'f1': f"{f1:.4f}",
    }


def evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test):
    """Evaluates a list of classifiers and returns their performance metrics.

    Parameters:
        classifiers (list): List of sklearn models to evaluate.
        X_train, X_test, y_train, y_test: Dataset splits.

    Returns:
        pd.DataFrame: A DataFrame with evaluation metrics for all classifiers.
    """
    results = [evaluate_classifier(clf, X_train, X_test, y_train, y_test) for clf in classifiers]
    results_df = pd.DataFrame(results)
    results_df.set_index('Classifier', inplace=True)
    return results_df


def plot_metrics(metrics_df):
    """Plots performance metrics for classifiers.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing performance metrics.
    """
    # Convert metrics to numeric
    metrics_df[['Accuracy', 'Precision', 'Recall', 'f1']] = metrics_df[
        ['Accuracy', 'Precision', 'Recall', 'f1']].apply(pd.to_numeric)

    # Plot each metric
    plt.figure(figsize=(10, 6))
    for column in ['Accuracy', 'Precision', 'Recall', 'f1']:
        plt.plot(metrics_df.index, metrics_df[column], marker='o', linestyle='-', label=column)

    # Add titles and labels
    plt.title('Classifiers Performance Metrics')
    plt.xlabel('Classifier')
    plt.ylabel('Metric Value')
    plt.ylim(0.2, 1.0)
    plt.legend(title='Metrics')
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Adjust pandas display settings
    adjust_pandas_display(max_rows=None, max_columns=None)

    # Load and preprocess dataset
    df = load_pkl("articles_vectorized_tfidf.pkl").dropna()
    print(f"Loaded DataFrame:\n{df.head()}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, "text", "category")

    # Define classifiers
    classifiers_to_evaluate = [
        LogisticRegression(solver='liblinear', max_iter=500),
        DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42),
        RandomForestClassifier(n_estimators=40, max_depth=30, bootstrap=True),
        KNeighborsClassifier(n_neighbors=8, metric='minkowski'),
    ]

    # Evaluate classifiers
    accuracy_df = evaluate_classifiers(classifiers_to_evaluate, X_train, X_test, y_train, y_test)

    # Print the accuracy table
    print("\nAccuracy Table for All Classifiers")
    print(accuracy_df)

    # Plot metrics
    plot_metrics(accuracy_df)
