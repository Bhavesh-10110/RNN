# RNN Practical (Many to one)
# SMS Spam Detection using Simple RNN

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50


# ------------------------------
# Clean Text
# ------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------
# Train Model
# ------------------------------
def train_model():
    print("Training model...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print("\nOriginal Labels:")
    print(df["label"].value_counts())

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    print("\nMapped Labels:")
    print(df["label"].value_counts())

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post"
    )

    y = df["label"].values

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    # Save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        0: class_weights[0],
        1: class_weights[1]
    }

    print("Class Weights:", class_weights)

    # Build model
    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        )
    )

    model.add(SimpleRNN(128))

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.summary()

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=32,
        class_weight=class_weights
    )

    # Save model
    model.save(MODEL)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)

    print("\nTest Accuracy:", accuracy)

    predictions = (
        model.predict(X_test) > 0.5
    ).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ------------------------------
# Predict SMS
# ------------------------------
def predict_sms(message):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]

    prediction = "Spam" if probability > 0.8 else "Ham"

    return prediction, probability


# ------------------------------
# Train if model does not exist
# ------------------------------
if not os.path.exists(MODEL):
    train_model()


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("SMS Spam Detection using RNN")
st.write("Many-to-One RNN Example")

message = st.text_area("Enter SMS Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction, probability = predict_sms(message)

        if prediction == "Spam":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.write(f"Spam Probability: {probability:.4f}")