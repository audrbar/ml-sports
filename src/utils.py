import ast
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def adjust_pandas_display(max_rows=None, max_columns=None, width=None):
    """Adjusts pandas display settings to control the number of rows, columns, and display width.

    Parameters:
        max_rows (int, optional): Maximum number of rows to display. Default is None (show all rows).
        max_columns (int, optional): Maximum number of columns to display. Default is None (show all columns).
        width (int): Width of the display in characters. Default is 1000.
    """
    pd.set_option('display.max_rows', max_rows)  # Show all rows or up to max_rows
    pd.set_option('display.max_columns', max_columns)  # Show all columns or up to max_columns
    pd.set_option('display.width', width)  # Adjust the display width
    pd.set_option('display.max_colwidth', None)  # Display full content of each cell
    print("\nPandas display settings adjusted.")


def write_csv(data_frame, output_file, index=False):
    """Writes a DataFrame to a CSV file.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to write to the CSV file.
        output_file (str): The name of the output CSV file.
        index (bool): Whether to include the index in the CSV file. Default is False.

    Returns:
        None
    """
    try:
        data_frame.to_csv(output_file, index=index)
        print(f"\nData successfully written to file '{output_file}'.")
    except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")


def append_csv(data, filename):
    """Appends the scraped data to a CSV file."""
    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Append to the file if it exists, otherwise create it
    if file_exists:
        df.to_csv(filename, mode='a', index=False, header=False)
        print(f"Data appended to file '{filename}'")
    else:
        df.to_csv(filename, index=False)
        print(f"Data saved to file '{filename}'")


def load_csv(filename):
    """Loads a CSV file into a DataFrame. Exits the program if the file is not found.

    Parameters:
        filename (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        data_frame = pd.read_csv(filename)
        print(f"\nFile '{filename}' successfully loaded.")
        return data_frame
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please check the path.")
        exit()


def write_pkl(data_frame, filename, file_format="pkl"):
    """Saves vectorized data in binary format (.pkl or .npy).

    Parameters:
        data_frame (pd.DataFrame): DataFrame containing the vectorized data to save.
        filename (str): The file path (without extension) to save the data.
        file_format (str): The format to save the data ('pkl' or 'npy'). Default is 'pkl'.

    Returns:
        None
    """
    try:
        if file_format == "pkl":
            # Save DataFrame as a pickle file
            with open(f"{filename}.pkl", "wb") as file:
                pickle.dump(data_frame, file)
            print(f"Data successfully saved to '{filename}.pkl'.")

        elif file_format == "npy":
            # Convert DataFrame to a NumPy array and save as .npy
            np_data = data_frame.to_numpy()
            np.save(f"{filename}.npy", np_data)
            print(f"Data successfully saved to '{filename}.npy'.")

        else:
            print("Error: Unsupported format. Please choose 'pkl' or 'npy'.")

    except Exception as e:
        print(f"An error occurred while saving the data: {e}")


def load_pkl(filename):
    """Reads a .pkl file and loads its content into a pandas DataFrame.

    Parameters:
        filename (str): The path to the .pkl file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        # Open the .pkl file and load its content
        with open(filename, "rb") as file:
            data_frame = pickle.load(file)
            print(f"File '{filename}' successfully loaded.")
            return data_frame
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def check_class_distribution(data, label_column):
    """Checks the class distribution in a dataset and prints the counts and percentages.

    Parameters:
        data (pd.DataFrame): The dataset containing the label column.
        label_column (str): The name of the column containing class labels.

    Returns:
        pd.DataFrame: A DataFrame showing class counts and percentages.
    """
    # Count the occurrences of each class
    class_counts = data[label_column].value_counts()

    # Calculate the percentage for each class
    class_percentages = data[label_column].value_counts(normalize=True) * 100

    # Combine counts and percentages into a DataFrame
    class_distribution = pd.DataFrame({
        'Count': class_counts,
        'Percentage': class_percentages
    }).sort_index()

    print(f"\nClass Distribution:\n{class_distribution}")

    return class_distribution


def split_data(data, text_column, label_column, test_size=0.2, val_size=0.2, random_state=42):
    """Splits the dataset into training, testing, and validation sets.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing text and label columns.
        text_column (str): The name of the column containing text data.
        label_column (str): The name of the column containing label data.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2 (20%).
        val_size (float): Proportion of the training data to include in the validation split. Default is 0.2 (20%).
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Extract features (X) and labels (y)
    X = np.array(data[text_column].tolist())  # Convert the column into a 2D array
    y = data[label_column]

    # Step 1: Split data into training+validation and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 2: Split training+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    # Output sizes for debugging
    print(f"Sizes of the data sets:")
    print(f"- training: {len(X_train)} samples")
    print(f"- validation: {len(X_val)} samples")
    print(f"- testing: {len(X_test)} samples")
    print(f"X shape: {X.shape}, type: {type(X)}")
    print(f"y shape: {y.shape}, type: {type(y)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
