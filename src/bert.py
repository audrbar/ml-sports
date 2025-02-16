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
from src.utils import adjust_pandas_display, load_csv, split_data_to_three, DATA_DIR, MODELS_DIR

# Define paths for storing data
ARTICLES_PREPROCESSED_CSV = os.path.join(DATA_DIR, "articles_preprocessed.csv")
BEST_BERT_MODEL = os.path.join(MODELS_DIR, "best_bert_model.keras")
BEST_LABEL_ENCODER = os.path.join(MODELS_DIR, "best_bert_label_encoder.pkl")

# Check library versions
print(f"TensorFlow Version: {tf.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")

# Load Hugging Face Accuracy Metric
accuracy_metric = evaluate.load("accuracy")

# Tokenizer Hyperparameter Tuning
MAX_LENGTH_OPTIONS = [128, 256, 512]
DROPOUT_RATES = [0.1, 0.3, 0.5]


def load_data(file_path):
    """Loads text data and encodes labels."""
    df = load_csv(file_path).dropna()
    X = df["text"].astype(str).tolist()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    return X, y, label_encoder


def tokenize_function(examples, max_length):
    """Tokenizes text using DistilBERT tokenizer with a specific max length."""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)


def prepare_dataset(X_train, X_val, X_test, y_train, y_val, y_test, max_length):
    """Converts datasets into Hugging Face DatasetDict and tokenizes them."""
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": X_train, "label": y_train}),
        "validation": Dataset.from_dict({"text": X_val, "label": y_val}),
        "test": Dataset.from_dict({"text": X_test, "label": y_test}),
    })
    return dataset.map(lambda x: tokenize_function(x, max_length=max_length), batched=True)


def compute_metrics(eval_pred):
    """Computes accuracy for the model."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


class CustomDistilBertModel(nn.Module):
    def __init__(self, num_labels, dropout=0.3):
        super(CustomDistilBertModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, max_length, dropout):
    """Trains and evaluates a DistilBERT model with given hyperparameters."""
    tokenized_datasets = prepare_dataset(X_train, X_val, X_test, y_train, y_val, y_test, max_length)
    model = CustomDistilBertModel(num_labels=len(set(y_train)), dropout=dropout)

    training_args = TrainingArguments(
        output_dir='./data',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate(tokenized_datasets["test"])

    return model, results["eval_accuracy"]


if __name__ == "__main__":
    adjust_pandas_display(max_rows=None, max_columns=None)
    X, y, label_encoder = load_data(ARTICLES_PREPROCESSED_CSV)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_to_three(X, y)

    results_df = pd.DataFrame(columns=["max_length", "dropout", "accuracy"])
    best_accuracy = 0
    best_model = None
    best_params = {}

    for max_length in MAX_LENGTH_OPTIONS:
        for dropout in DROPOUT_RATES:
            print(f"Training with max_length={max_length}, dropout={dropout}")
            model, accuracy = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, max_length, dropout)
            print(f"Accuracy: {accuracy:.4f}")

            # Store results
            new_result = pd.DataFrame({"max_length": [max_length], "dropout": [dropout], "accuracy": [accuracy]})
            results_df = pd.concat([results_df, new_result], ignore_index=True)

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_params = {"max_length": max_length, "dropout": dropout}

    print("\nCustom Distil Bert Model Final Hyperparameter Tuning Results:")
    print(results_df)

    print(f"\nCustom Distil Bert Model Best Model Parameters: {best_params}, Accuracy: {best_accuracy:.4f}")

    # Save the best model
    if best_model:
        torch.save(best_model.state_dict(), BEST_BERT_MODEL)
        print(f"Best model saved to {BEST_BERT_MODEL}")

        # Save label encoder for consistency in inference
        with open(BEST_LABEL_ENCODER, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to {BEST_LABEL_ENCODER}")

# import os
# import tensorflow as tf
# import transformers
# import torch
# import evaluate
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import pickle
# from datasets import Dataset, DatasetDict
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from transformers import (DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments,
#                           AutoModelForSequenceClassification)
# from src.utils import adjust_pandas_display, load_csv, split_data_to_three, DATA_DIR, MODELS_DIR
#
# # Define paths for storing data
# ARTICLES_PREPROCESSED_CSV = os.path.join(DATA_DIR, "articles_preprocessed.csv")
# BERT_MODEL = os.path.join(MODELS_DIR, "bert_trained_model.keras")
# BERT_ENCODER = os.path.join(MODELS_DIR, "bert_label_encoder.pkl")
#
# # Check library versions
# print("TensorFlow version:", tf.__version__)
# print(f"PyTorch Version: {torch.__version__}")
# print(f"Transformers Version: {transformers.__version__}")
#
# # Load Hugging Face Accuracy Metric
# accuracy_metric = evaluate.load("accuracy")
#
# # Tokenizer Hyperparameter Tuning
# MAX_LENGTH_OPTIONS = [128, 256, 512]
# DROPOUT_RATES = [0.1, 0.3, 0.5]
#
#
# def load_data(file_path):
#     """Loads text data and encodes labels."""
#     df = load_csv(file_path).dropna()
#     X = df["text"].astype(str).tolist()
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["category"])
#     return X, y, label_encoder
#
#
# def tokenize_function(examples, max_length):
#     """Tokenizes text using DistilBERT tokenizer with a specific max length."""
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
#
#
# def prepare_dataset(X_train, X_val, X_test, y_train, y_val, y_test, max_length):
#     """Converts datasets into Hugging Face DatasetDict and tokenizes them."""
#     dataset = DatasetDict({
#         "train": Dataset.from_dict({"text": X_train, "label": y_train}),
#         "validation": Dataset.from_dict({"text": X_val, "label": y_val}),
#         "test": Dataset.from_dict({"text": X_test, "label": y_test}),
#     })
#     return dataset.map(lambda x: tokenize_function(x, max_length=max_length), batched=True)
#
#
# def compute_metrics(eval_pred):
#     """Computes accuracy for the model."""
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return accuracy_metric.compute(predictions=predictions, references=labels)
#
#
# class CustomDistilBertModel(nn.Module):
#     def __init__(self, num_labels, dropout=0.3):
#         super(CustomDistilBertModel, self).__init__()
#         self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
#         self.classifier = nn.Sequential(
#             nn.Linear(self.distilbert.config.hidden_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_labels)
#         )
#         self.loss_fn = nn.CrossEntropyLoss()
#
#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
#         logits = self.classifier(outputs.last_hidden_state[:, 0, :])
#         loss = self.loss_fn(logits, labels) if labels is not None else None
#         return {"loss": loss, "logits": logits}
#
#
# def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, max_length, dropout):
#     """Trains and evaluates a DistilBERT model with given hyperparameters."""
#     tokenized_datasets = prepare_dataset(X_train, X_val, X_test, y_train, y_val, y_test, max_length)
#     model = CustomDistilBertModel(num_labels=len(set(y_train)), dropout=dropout)
#     training_args = TrainingArguments(
#         output_dir='./data',
#         eval_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_total_limit=2,
#         save_strategy="epoch",
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_datasets['train'],
#         eval_dataset=tokenized_datasets['validation'],
#         compute_metrics=compute_metrics
#     )
#     trainer.train()
#     results = trainer.evaluate(tokenized_datasets["test"])
#     return results["eval_accuracy"]
#
#
# if __name__ == "__main__":
#     adjust_pandas_display(max_rows=None, max_columns=None)
#     X, y, label_encoder = load_data(ARTICLES_PREPROCESSED_CSV)
#     X_train, X_val, X_test, y_train, y_val, y_test = split_data_to_three(X, y)
#
#     results_df = pd.DataFrame(columns=["max_length", "dropout", "accuracy"])
#
#     for max_length in MAX_LENGTH_OPTIONS:
#         for dropout in DROPOUT_RATES:
#             print(f"Training with max_length={max_length}, dropout={dropout}")
#             accuracy = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, max_length, dropout)
#             print(f"Accuracy: {accuracy:.4f}")
#             results_df = results_df.append({"max_length": max_length, "dropout": dropout, "accuracy": accuracy},
#                                            ignore_index=True)
#
#     print("Custom Distil Bert Model Final Hyperparameter Tuning Results:")
#     print(results_df)
#
#     best_params = results_df.loc[results_df['accuracy'].idxmax()]
#     print(f"Best Custom Distil Bert Model Parameters: {best_params.to_dict()}")
#
# # Training with max_length=128, dropout=0.1 Accuracy: 0.9438
# # Training with max_length=128, dropout=0.3 Accuracy: 0.9270
# # Training with max_length=128, dropout=0.5 Accuracy: 0.9213
# # Training with max_length=256, dropout=0.1 Accuracy: 0.9663
# # Training with max_length=256, dropout=0.3 Accuracy: 0.9551
# # Training with max_length=256, dropout=0.5 Accuracy: 0.9494
# # Training with max_length=512, dropout=0.1 Accuracy: 0.9607
# # Training with max_length=512, dropout=0.3 Accuracy: 0.9551
# # Training with max_length=512, dropout=0.5 Accuracy: 0.9551
# # Best Parameters: {'max_length': 256, 'dropout': 0.1}, Best Accuracy: 0.9663
