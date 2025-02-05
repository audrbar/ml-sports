# import tensorflow as tf
# import transformers
# import torch
#
# print(f"PyTorch Version: {torch.__version__}")
# print(f"Transformers Version: {transformers.__version__}")
# print(f"Using MPS (Metal): {torch.backends.mps.is_available()}")
# Load sentiment analysis model
# classifier = pipeline("sentiment-analysis")
#
# # Test inference
# result = classifier("I love using Transformers on Apple Silicon!")
# print("Classified:\n", result)
# # Check if TensorFlow is available
# print("TensorFlow version:", tf.__version__)
#
# # Check if MPS (Metal Performance Shaders) backend is available
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("MPS (Metal Performance Shaders) backend is available.")
#     print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# else:
#     print("No GPU found. Running on CPU.")

# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)
#
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=8)

# import torch
# x = torch.rand(5, 3)
# print(x)
# import tensorflow as tf
#
# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)
#
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=2, batch_size=16)

# pip install tensorflow
# pip install tensorflow-macos
# pip install tensorflow-metal

import torch
import transformers
from tensorflow import keras
from transformers import pipeline
#
#
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ✅ Step 1: Load DistilBERT from TensorFlow Hub
pretrained_model = "https://tfhub.dev/tensorflow/distilbert_en_uncased_preprocess/3"
bert_encoder = "https://tfhub.dev/tensorflow/distilbert_en_uncased_L-6_H-768_A-12/2"

preprocessor = hub.KerasLayer(pretrained_model, name="preprocessing")
encoder = hub.KerasLayer(bert_encoder, trainable=True, name="BERT_encoder")


# ✅ Step 2: Load and Prepare Data
def load_data():
    # Example dataset
    df = pd.DataFrame({
        "text": [
            "Apple releases new MacBook Pro!",
            "Manchester United wins the Champions League.",
            "NASA discovers a new exoplanet.",
            "Bitcoin price reaches new all-time high.",
        ],
        "category": ["technology", "sports", "science", "finance"]
    })

    label_encoder = LabelEncoder()
    df["category_encoded"] = label_encoder.fit_transform(df["category"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category_encoded"], test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoder


X_train, X_test, y_train, y_test, label_encoder = load_data()


# ✅ Step 3: Preprocess Data for DistilBERT
def encode_text(texts):
    return preprocessor(tf.constant(texts))


train_encodings = encode_text(X_train.tolist())
test_encodings = encode_text(X_test.tolist())


# ✅ Step 4: Build DistilBERT Classification Model
def build_model(num_classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = preprocessor(text_input)
    encoder_output = encoder(preprocessing_layer)["pooled_output"]

    # Classification head
    dropout = tf.keras.layers.Dropout(0.3)(encoder_output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(dropout)

    model = tf.keras.Model(inputs=text_input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = build_model(num_classes=len(label_encoder.classes_))

# ✅ Step 5: Train the Model
model.fit(
    x=X_train.tolist(), y=y_train,
    validation_data=(X_test.tolist(), y_test),
    epochs=3, batch_size=8
)

# ✅ Step 6: Evaluate the Model
loss, accuracy = model.evaluate(X_test.tolist(), y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# # ✅ Step 7: Save Model
# model.save("distilbert_macos.h5")
