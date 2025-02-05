import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import DATA_DIR

# Define paths for storing data
ARTICLES_PREPROCESSED_CSV = os.path.join(DATA_DIR, "articles_preprocessed.csv")
ARTICLES_VECTORIZED_WORD2VEC = os.path.join(DATA_DIR, "articles_vectorized_word2vec.pkl")


def load_data_pkl(file_path):
    """Loads vectorized text features and labels from a pickle file.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        tuple: (X, y, label_encoder)
    """
    df = pd.read_pickle(file_path).dropna()

    # Convert 'text' column (vectorized features) into a NumPy array
    X = np.array(df["text"].tolist(), dtype=np.float32)

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    y = to_categorical(y)  # Convert labels to one-hot encoding

    return X, y, label_encoder


def load_data(file_path):
    """Loads text features and labels from a csv file.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        tuple: (X, y, label_encoder)
    """
    df = pd.read_csv(file_path).dropna()

    # Remove non-UTF-8 characters
    df['text'] = df['text'].apply(lambda x: x.encode('ascii', 'ignore').decode('utf-8'))

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])

    # # Convert 'text' column (vectorized features) into a NumPy array
    # X = np.array(df["text"].tolist(), dtype=np.float32)

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
    # Add an embedding layer to the model
    model = Sequential([
        Input(shape=(input_dim, 1)),  # Reshaped for LSTM
        LSTM(128, return_sequences=True),  # First LSTM Layer
        LSTM(64),  # Second LSTM Layer
        Dense(128, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
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
    """Evaluates the trained model on the test dataset.

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
    # # Load vectorized dataset
    # X, y, label_encoder = load_data(ARTICLES_PREPROCESSED_CSV)

    # Load vectorized dataset
    X, y, label_encoder = load_data_pkl(ARTICLES_VECTORIZED_WORD2VEC)

    # # Reshape X to fit LSTM layer (RNN requires 3D input: (samples, time steps, features))
    # X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # # Balance classes using SMOTE (Synthetic Minority Oversampling)
    # smote = SMOTE()
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    # Build RNN model
    rnn_model = build_rnn_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    # Train RNN model
    train_model(rnn_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64)

    # Evaluate RNN model
    evaluate_model(rnn_model, X_test, y_test)

    # # Save the trained model & encoder
    # save_model_and_encoder(rnn_model, label_encoder)
