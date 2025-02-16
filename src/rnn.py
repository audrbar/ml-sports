import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from utils import DATA_DIR, MODELS_DIR

# Define paths for storing data
DATA_FILES = {
    "tfidf": os.path.join(DATA_DIR, "articles_vectorized_tfidf.pkl"),
    "word2vec": os.path.join(DATA_DIR, "articles_vectorized_word2vec.pkl"),
    "ngram": os.path.join(DATA_DIR, "articles_vectorized_ngram.pkl"),
    "FastText": os.path.join(DATA_DIR, "articles_vectorized_fasttext.pkl"),
}
RNN_MODEL = os.path.join(MODELS_DIR, "rnn_trained_model.keras")
RNN_ENCODER = os.path.join(MODELS_DIR, "rnn_label_encoder.pkl")


def load_data_pkl(file_path):
    df = pd.read_pickle(file_path).dropna()
    X = np.array(df["text"].tolist(), dtype=np.float32)
    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(df["category"]))
    return X, y, label_encoder


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                                stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                      random_state=random_state, stratify=y_train_val)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_rnn_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim, 1)),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(128, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(X, y, file_key):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    model = build_rnn_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, verbose=1)
    test_accuracy = model.evaluate(X_test, y_test)[1]
    return test_accuracy, model


if __name__ == "__main__":
    results = {}
    for key, file_path in DATA_FILES.items():
        print(f"Processing {key} dataset...")
        X, y, label_encoder = load_data_pkl(file_path)
        test_accuracy, model = train_and_evaluate_model(X, y, key)
        results[key] = test_accuracy

    best_model_key = max(results, key=results.get)
    print(f"Best model: {best_model_key} with accuracy: {results[best_model_key]:.4f}")

    with open(RNN_ENCODER, "wb") as f:
        pickle.dump(label_encoder, f)
    model.save(RNN_MODEL)
    print(model.summary())
    print("Model and encoder saved.")

# Best model: FastText with accuracy: 0.4438
