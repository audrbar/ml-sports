import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils import adjust_pandas_display, load_pkl, DATA_DIR, MODELS_DIR

# Define paths for storing data
DATA_FILES = {
    "tfidf": os.path.join(DATA_DIR, "articles_vectorized_tfidf.pkl"),
    "word2vec": os.path.join(DATA_DIR, "articles_vectorized_word2vec.pkl"),
    "ngram": os.path.join(DATA_DIR, "articles_vectorized_ngram.pkl"),
}
SNN_MODEL = os.path.join(DATA_DIR, "snn_trained_model.keras")
SNN_ENCODER = os.path.join(DATA_DIR, "snn_label_encoder.pkl")


def encode_labels(data_frame, features_column, labels_column):
    """Encodes text features into numerical format and encodes labels."""
    X = np.array(data_frame[features_column].tolist(), dtype=np.float32)
    label_encoder_ = LabelEncoder()
    y = label_encoder_.fit_transform(data_frame[labels_column])
    y = to_categorical(y)
    print(f"Final X shape: {X.shape}")
    return X, y, label_encoder_


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Splits the dataset into training, validation, and test sets."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state)
    print(f"Data split: Training={X_train.shape[0]}, Validation={X_val.shape[0]}, Testing={X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_dim, output_dim, neurons=[512, 256], dropout_rates=[0.3, 0.3]):
    """Builds a neural network model dynamically."""
    model_seq = Sequential([Input(shape=(input_dim,))])
    for n, d in zip(neurons, dropout_rates):
        model_seq.add(Dense(n, activation='relu'))
        model_seq.add(Dropout(d))
    model_seq.add(Dense(output_dim, activation='softmax'))
    model_seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_seq


def train_model(model_seq, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """Trains the neural network model."""
    history = model_seq.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def evaluate_model(model_seq, X_test, y_test):
    """Evaluates the model on the test dataset."""
    test_loss, test_accuracy = model_seq.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy


def save_model_and_encoder(model_seq, label_encoder, model_path, encoder_path):
    """Saves the trained model and label encoder."""
    model_seq.save(model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")


def plot_hyperparameter_tuning(results_df):
    """Plots the validation accuracy from hyperparameter tuning and includes configuration in legend."""
    plt.figure(figsize=(10, 6))
    for idx, row in results_df.iterrows():
        label = f"Neurons: {row['neurons']}, Dropout: {row['dropout_rates']}, Batch: {row['batch_size']}"
        plt.plot(idx, row['val_accuracy'], marker='o', linestyle='-', label=label)

    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("Validation Accuracy")
    plt.title("Hyperparameter Tuning Results")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    plt.show()


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Performs hyperparameter tuning for neural network architecture and saves results."""
    tuning_results = []
    neurons_list = [[512, 256], [1024, 512], [256, 128]]
    dropout_rates_list = [[0.3, 0.3], [0.5, 0.5], [0.2, 0.2]]
    batch_sizes = [32, 64]
    epochs = 50

    for neurons in neurons_list:
        for dropout_rates in dropout_rates_list:
            for batch_size in batch_sizes:
                print(f"Training with neurons={neurons}, dropout_rates={dropout_rates}, batch_size={batch_size}")
                model = build_model(X_train.shape[1], y_train.shape[1], neurons=neurons, dropout_rates=dropout_rates)
                history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                val_acc = max(history.history['val_accuracy'])

                tuning_results.append({
                    'neurons': neurons,
                    'dropout_rates': dropout_rates,
                    'batch_size': batch_size,
                    'val_accuracy': val_acc
                })

    results_df = pd.DataFrame(tuning_results)
    results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
    print("Hyperparameter tuning results saved.")

    plot_hyperparameter_tuning(results_df)

    best_row = results_df.loc[results_df['val_accuracy'].idxmax()]
    print(f"Best parameters: {best_row}")
    return {
        'neurons': best_row['neurons'],
        'dropout_rates': best_row['dropout_rates'],
        'batch_size': best_row['batch_size']
    }


if __name__ == "__main__":
    adjust_pandas_display(max_rows=None, max_columns=None)
    df = load_pkl(ARTICLES_VECTORIZED_TFIDF).dropna()
    X, y, label_encoder = encode_labels(df, features_column="text", labels_column="category")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    model = build_model(X_train.shape[1], y_train.shape[1], neurons=best_params['neurons'],
                        dropout_rates=best_params['dropout_rates'])
    train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=best_params['batch_size'])
    evaluate_model(model, X_test, y_test)
    save_model_and_encoder(model, label_encoder, SNN_MODEL, SNN_ENCODER)

    # Best hyperparameters: {'neurons': [512, 256], 'dropout_rates': [0.2, 0.2], 'batch_size': 32},
    # Validation Accuracy: 0.9318
    # Test Accuracy: 0.9236
