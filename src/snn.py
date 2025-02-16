import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils import adjust_pandas_display, load_pkl, split_data_to_three, DATA_DIR, MODELS_DIR

# Define paths for storing data
DATA_FILES = {
    "tfidf": os.path.join(DATA_DIR, "articles_vectorized_tfidf.pkl"),
    "word2vec": os.path.join(DATA_DIR, "articles_vectorized_word2vec.pkl"),
    "ngram": os.path.join(DATA_DIR, "articles_vectorized_ngram.pkl"),
    "FastText": os.path.join(DATA_DIR, "articles_vectorized_fasttext.pkl"),
}
SNN_MODEL = os.path.join(MODELS_DIR, "snn_trained_model.keras")
SNN_ENCODER = os.path.join(MODELS_DIR, "snn_label_encoder.pkl")


def encode_labels(df, features_col, labels_col):
    """Encodes text features into numerical format and encodes labels."""
    X = np.vstack(df[features_col].values).astype(np.float32)
    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(df[labels_col]))
    print(f"Data Shape - Features: {X.shape}, Labels: {y.shape}")
    return X, y, label_encoder


def build_model(input_dim, output_dim, neurons=[512, 256], dropout_rates=[0.3, 0.3]):
    """Builds a flexible feedforward neural network."""
    model = Sequential([Input(shape=(input_dim,))])
    for n, d in zip(neurons, dropout_rates):
        model.add(Dense(n, activation='relu'))
        model.add(Dropout(d))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    """Trains the model with early stopping."""
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stop]
    )
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def save_model_and_encoder(model, label_encoder, model_path, encoder_path):
    """Saves the trained model and label encoder."""
    model.save(model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Model saved to {model_path}, Label encoder saved to {encoder_path}")


def plot_hyperparameter_tuning(results_df):
    """Plots validation accuracy from hyperparameter tuning."""
    plt.figure(figsize=(10, 6))
    for idx, row in results_df.iterrows():
        label = f"Neurons: {row['neurons']}, Dropout: {row['dropout_rates']}, Batch: {row['batch_size']}"
        plt.plot(idx, row['val_accuracy'], marker='o', linestyle='-', label=label)
    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("Validation Accuracy")
    plt.title("Tensorflow Keras Sequential Model Hyperparameter Tuning Results")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    plt.show()


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Performs hyperparameter tuning and saves results."""
    tuning_results = []
    neurons_list = [[512, 256], [1024, 512], [256, 128]]
    dropout_rates_list = [[0.3, 0.3], [0.5, 0.5], [0.2, 0.2]]
    batch_sizes = [32, 64]
    epochs = 100

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
    print(f"Tensorflow Keras Sequential Model Hyperparameter Tuning Results:\n{results_df}")
    plot_hyperparameter_tuning(results_df)
    best_row = results_df.loc[results_df['val_accuracy'].idxmax()]
    print(f"Best Tensorflow Keras Sequential Model Hyperparameter Configuration:\n{best_row.to_dict()}")
    return {'neurons': best_row['neurons'], 'dropout_rates': best_row['dropout_rates'],
            'batch_size': best_row['batch_size']}


if __name__ == "__main__":
    adjust_pandas_display(max_rows=None, max_columns=None)
    df = load_pkl(DATA_FILES["tfidf"]).dropna()
    X, y, label_encoder = encode_labels(df, features_col="text", labels_col="category")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_to_three(X, y)
    best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    model = build_model(X_train.shape[1], y_train.shape[1], neurons=best_params['neurons'],
                        dropout_rates=best_params['dropout_rates'])
    print(model.summary())
    train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=best_params['batch_size'])
    evaluate_model(model, X_test, y_test)
    save_model_and_encoder(model, label_encoder, SNN_MODEL, SNN_ENCODER)

# Best Hyperparameter Configuration:
# {'neurons': [512, 256], 'dropout_rates': [0.3, 0.3], 'batch_size': 32, 'val_accuracy': 0.9867549538612366}
# Test Accuracy: 0.9787
