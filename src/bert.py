import os
import tensorflow as tf
import transformers
import torch
import evaluate
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments,
                          AutoModelForSequenceClassification)
from src.utils import adjust_pandas_display, DATA_DIR

# Define paths for storing data
ARTICLES_VECTORIZED_TFIDF = os.path.join(DATA_DIR, "articles_vectorized_tfidf.pkl")

# https://ajeygore.in/content/TensorFlow2-on-macOS-M1-Machines

# Check if packets are available
print("TensorFlow version:", tf.__version__)
print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"Using MPS (Metal Performance Shaders): {torch.backends.mps.is_available()}")

# Check if MPS (Metal Performance Shaders) backend is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("MPS (Metal Performance Shaders) backend is available.")
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPU found. Running on CPU.")

# Load Hugging Face Accuracy Metric
accuracy_metric = evaluate.load("accuracy")


def load_data(file_path):
    """Loads vectorized text features and labels from a pickle file.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        tuple: (X, y, label_encoder)
    """
    df_r = pd.read_pickle(file_path).dropna()

    # Convert text column to list
    X = df_r["text"].astype(str).tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_r["category"])

    return X, y, label_encoder


# Split Data into Training, Validation, and Test Sets
def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Splits the dataset into training, validation, and test sets.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    print(f"Data split: Training={len(X_train)}, Validation={len(X_val)}, Testing={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# Tokenization Function
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokenize_function(examples):
    """Tokenizes the text input using DistilBERT tokenizer.

    Parameters:
        examples (dict): Dictionary containing 'text' data.

    Returns:
        dict: Tokenized text with 'input_ids' and 'attention_mask'.
    """
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)


def prepare_dataset(X_train, X_val, X_test, y_train, y_val, y_test):
    """Converts datasets into Hugging Face DatasetDict and tokenizes them."""
    # Convert data to lists to avoid format issues
    X_train, X_val, X_test = map(lambda x: list(map(str, x)), [X_train, X_val, X_test])
    y_train, y_val, y_test = map(list, [y_train, y_val, y_test])

    # Create Hugging Face DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": X_train, "label": y_train}),
        "validation": Dataset.from_dict({"text": X_val, "label": y_val}),
        "test": Dataset.from_dict({"text": X_test, "label": y_test}),
    })

    # Tokenize dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets


# Compute Metrics Function
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy for the model.

    Parameters:
        eval_pred: Evaluation predictions.

    Returns:
        dict: Accuracy score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


# Custom Model
class CustomDistilBertModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomDistilBertModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)  # Use num_labels dynamically
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with optional label handling for Hugging Face Trainer."""
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])

        # If labels are provided, calculate loss (needed for Hugging Face Trainer)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# Training Arguments
training_args = TrainingArguments(
    output_dir='./data',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    save_strategy="epoch"
)

if __name__ == "__main__":
    # Adjust pandas display settings
    adjust_pandas_display(max_rows=None, max_columns=None)

    # Load dataset
    X, y, label_encoder = load_data(ARTICLES_VECTORIZED_TFIDF)
    print(f"Loaded DataFrame:\n{X[:5]}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(type(X_train))  # Should be list
    print(X_train[:5])  # Should display text samples

    # Convert to Hugging Face DatasetDict and tokenize
    tokenized_datasets = prepare_dataset(X_train, X_val, X_test, y_train, y_val, y_test)

    # Load pre-trained DistilBERT model
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "distilbert-base-uncased", num_labels=len(set(y_train))
    # )
    num_labels = len(set(y_train))
    model = CustomDistilBertModel(num_labels=num_labels)

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train Model
    print("Training started...")
    trainer.train()

    # Evaluate Model
    print("Evaluating model...")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")

    # # Save Model & Label Encoder
    # model.save_pretrained("distilbert_finetuned")
    # tokenizer.save_pretrained("distilbert_finetuned")
    # with open("label_encoder.pkl", "wb") as f:
    #     pickle.dump(label_encoder, f)
    # print("Model and tokenizer saved successfully!")
