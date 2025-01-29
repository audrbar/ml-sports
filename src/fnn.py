import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils import adjust_pandas_display, load_pkl


def encode_labels(data_frame, features_column, labels_column):
    """Encodes text features into numerical format and encodes labels.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame.
        features_column (str): Column name for text features.
        labels_column (str): Column name for labels.

    Returns:
        tuple: (X, y, label_encoder) where:
            - X: 2D NumPy array of vectorized text features.
            - y: One-hot encoded array of labels.
            - label_encoder: The fitted LabelEncoder instance.
    """
    X = np.array(data_frame[features_column].tolist(), dtype=np.float32)  # Convert text features to NumPy array
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data_frame[labels_column])  # Convert labels to integers
    y = to_categorical(y)  # Convert labels to one-hot encoding
    print(f"Final X shape: {X.shape}")  # Should be (n_samples, n_features)
    return X, y, label_encoder


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
        X (np.array): Feature matrix.
        y (np.array): Label matrix.
        test_size (float): Proportion of data for testing.
        val_size (float): Proportion of training data for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state)

    print(f"Data split: Training={X_train.shape[0]}, Validation={X_val.shape[0]}, Testing={X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_dim, output_dim):
    """Builds a simple neural network model for multi-class classification.

    Parameters:
        input_dim (int): Number of input features.
        output_dim (int): Number of output categories.

    Returns:
        tf.keras.Model: The compiled model.
    """
    model_seq = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),  # Input layer
        Dropout(0.3),  # Regularization
        Dense(256, activation='relu'),  # Hidden layer
        Dropout(0.3),
        Dense(output_dim, activation='softmax')  # Output layer for multi-class classification
    ])

    model_seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_seq


def train_model(model_seq, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """Trains the neural network model.

    Parameters:
        model_seq (tf.keras.Model): The compiled model.
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size.

    Returns:
        tf.keras.callbacks.History: The training history.
    """
    history = model_seq.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def evaluate_model(model_seq, X_test, y_test):
    """Evaluates the model on the test dataset.

    Parameters:
        model_seq (tf.keras.Model): The trained model.
        X_test (np.array): Test features.
        y_test (np.array): Test labels.

    Returns:
        float: Test accuracy.
    """
    test_loss, test_accuracy = model_seq.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy


def save_model_and_encoder(model_seq, labels_encoder, model_path="snn_classification_model.keras",
                           encoder_path="label_encoder.pkl"):
    """Saves the trained model and label encoder.

    Parameters:
        model_seq (tf.keras.Model): The trained model.
        labels_encoder (LabelEncoder): The fitted label encoder.
        model_path (str): Path to save the model.
        encoder_path (str): Path to save the label encoder.
    """
    model_seq.save(model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(labels_encoder, f)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")


if __name__ == "__main__":
    # Adjust pandas display settings
    adjust_pandas_display(max_rows=None, max_columns=None)

    # Load and preprocess dataset
    df = load_pkl("articles_vectorized_tfidf.pkl").dropna()

    # Encode features and labels
    X, y, label_encoder = encode_labels(df, features_column="text", labels_column="category")

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Build model dynamically based on dataset
    model = build_model(X_train.shape[1], y_train.shape[1])

    # Train the model
    train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save model and encoder
    save_model_and_encoder(model, label_encoder)
