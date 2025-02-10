import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read base source directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))


def adjust_pandas_display(max_rows=None, max_columns=None, width=None):
    """Adjust pandas display settings."""
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', width)
    pd.set_option('display.max_colwidth', None)


def write_csv(data_frame, output_file, index=False):
    """Writes DataFrame to CSV."""
    try:
        data_frame.to_csv(output_file, index=index)
    except Exception as e:
        print(f"Error writing to CSV: {e}")


def append_csv(data, filename: str):
    """Appends data to CSV file."""
    df = pd.DataFrame(data)
    mode, header = ('a', False) if os.path.isfile(filename) else ('w', True)
    df.to_csv(filename, mode=mode, index=False, header=header)


def load_csv(filename):
    """Loads CSV into DataFrame."""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None


def write_pkl(data, filename, file_format="pkl"):
    """Saves data in binary format."""
    try:
        if file_format == "pkl":
            with open(f"{filename}.pkl", "wb") as file:
                pickle.dump(data, file)
        elif file_format == "npy":
            np.save(f"{filename}.npy", np.array(data))
    except Exception as e:
        print(f"Error saving file: {e}")


def load_pkl(filename):
    """Loads data from pickle file."""
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def split_data_to_two(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def split_data_to_three(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Splits data into training, validation, and test sets."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_unique_values(first_file, second_file, output_file):
    """Finds and saves unique values from second CSV not in first CSV."""
    try:
        df_first = pd.read_csv(first_file) if os.path.exists(first_file) else pd.DataFrame()
        df_second = pd.read_csv(second_file)
        df_unique = df_second[~df_second.apply(tuple, axis=1).isin(df_first.apply(tuple, axis=1))]
        df_unique.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error processing unique values: {e}")
