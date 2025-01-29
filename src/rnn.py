import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle


def load_data(file_path):
    """Loads vectorized text features and labels from a pickle file.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        tuple: (X, y, label_encoder)
    """
    df = pd.read_pickle(file_path)

    # Convert 'text' column (vectorized features) into a NumPy array
    X = np.array(df["text"].tolist(), dtype=np.float32)

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    y = to_categorical(y)  # Convert labels to one-hot encoding

    return X, y, label_encoder


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Splits the dataset into training, validation, and test sets.

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
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )

    print(f"Data split: Training={X_train.shape[0]}, Validation={X_val.shape[0]}, Testing={X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_rnn_model(input_dim, output_dim):
    """Builds a Recurrent Neural Network (RNN) using LSTM for text classification.

    Parameters:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.

    Returns:
        tf.keras.Model: The compiled RNN model.
    """
    model = Sequential([
        Input(shape=(input_dim, 1)),  # Reshaped for LSTM
        LSTM(128, return_sequences=True),  # First LSTM Layer
        LSTM(64),  # Second LSTM Layer
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """Trains the RNN model.

    Parameters:
        model (tf.keras.Model): The compiled model.
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        tf.keras.callbacks.History: Training history.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test dataset.

    Parameters:
        model (tf.keras.Model): The trained model.
        X_test (np.array): Test features.
        y_test (np.array): Test labels.

    Returns:
        float: Test accuracy.
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy


def save_model_and_encoder(model, label_encoder, model_path="rnn_classification_model.keras", encoder_path="label_encoder.pkl"):
    """Saves the trained RNN model and label encoder.

    Parameters:
        model (tf.keras.Model): The trained model.
        label_encoder (LabelEncoder): The fitted label encoder.
        model_path (str): Path to save the model.
        encoder_path (str): Path to save the label encoder.
    """
    model.save(model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")


def predict_category(model, text_vectorized, label_encoder):
    """Predicts the category for a given vectorized text input.

    Parameters:
        model (tf.keras.Model): The trained model.
        text_vectorized (np.array): Vectorized input text.
        label_encoder (LabelEncoder): The label encoder.

    Returns:
        str: Predicted category.
    """
    prediction = model.predict(np.array([text_vectorized]).reshape(1, -1, 1))
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]


if __name__ == "__main__":
    # Load vectorized dataset
    X, y, label_encoder = load_data("articles_vectorized_ngram.pkl")

    # Reshape X to fit LSTM layer (RNN requires 3D input: (samples, time steps, features))
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Build RNN model
    rnn_model = build_rnn_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    # Train RNN model
    train_model(rnn_model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32)

    # Evaluate RNN model
    evaluate_model(rnn_model, X_test, y_test)

    # Save the trained model & encoder
    save_model_and_encoder(rnn_model, label_encoder)
