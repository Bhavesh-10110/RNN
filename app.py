# ===========================================================
# RNN Practical (Many-to-One)
# SMS Spam Detection using Simple RNN
# Streamlit Frontend
# ===========================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ===========================================================
# Constants
# ===========================================================

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50

DATASET_PATH = "spam.csv"


# ===========================================================
# Helpers
# ===========================================================

def clean_text(text):
    """
    Cleans raw SMS message text by lowercasing, removing punctuation, 
    and consolidating whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ===========================================================
# Train Model
# ===========================================================

def train_model(status=None, log=None):
    """
    Trains the Simple RNN Spam/Ham classifier.

    status : a st.status(...) context object used to show progress
    log    : a st.empty() placeholder used to stream text logs
    """

    def emit(msg):
        print(msg)
        if log is not None:
            log.markdown(msg)

    emit("Loading Dataset...")

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    df = pd.read_csv(DATASET_PATH, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    emit("Dataset Loaded Successfully!")

    # -------------------------------------------------------
    # Data Exploration
    # -------------------------------------------------------

    if status is not None:
        status.write("Exploring dataset...")

    with st.expander("Dataset Preview & Info", expanded=False):
        st.write("**First 5 Rows**")
        st.dataframe(df.head())

        st.write("**Dataset Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.write("**Class Counts:**")
        st.write(df["label"].value_counts())
        st.write("**Missing Values**")
        st.write(df.isnull().sum())

    # -------------------------------------------------------
    # Data Cleaning & Label Mapping
    # -------------------------------------------------------

    if status is not None:
        status.write("Cleaning dataset and mapping labels...")

    emit("Cleaning Dataset...")

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df["text"] = df["text"].apply(clean_text)

    emit("Cleaning Completed!")

    # -------------------------------------------------------
    # Text Encoding
    # -------------------------------------------------------

    if status is not None:
        status.write("Encoding words...")

    emit("Encoding Words...")

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["text"])

    X = tokenizer.texts_to_sequences(df["text"])
    y = df["label"].values

    emit(f"Vocabulary Size : {len(tokenizer.word_index)}")

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    emit("Tokenizer Saved Successfully!")

    # -------------------------------------------------------
    # Padding
    # -------------------------------------------------------

    if status is not None:
        status.write("Padding sequences...")

    emit("Padding Sequences...")

    X = pad_sequences(
        X,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    emit(f"Shape of X : {X.shape} &nbsp;&nbsp; Shape of y : {y.shape}")

    # -------------------------------------------------------
    # Train-Test Split & Class Weights
    # -------------------------------------------------------

    if status is not None:
        status.write("Splitting dataset & computing class weights...")

    emit("Splitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    emit(f"Training Samples : {X_train.shape[0]} &nbsp;&nbsp; Testing Samples : {X_test.shape[0]}")

    # Handle class weights due to dataset imbalance
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {
        0: class_weights_arr[0],
        1: class_weights_arr[1]
    }
    emit(f"Computed Class Weights: {class_weights}")

    # -------------------------------------------------------
    # Build Simple RNN
    # -------------------------------------------------------

    if status is not None:
        status.write("Building model...")

    emit("Building Model...")

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(
            128,
            return_sequences=False
        )
    )

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

    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))

    with st.expander("Model Summary", expanded=False):
        st.code("\n".join(summary_lines))

    # -------------------------------------------------------
    # Compile Model
    # -------------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------------------------------------
    # Train Model
    # -------------------------------------------------------

    if status is not None:
        status.write("Training model... this may take a while")

    emit("Training Model...")

    epochs = st.session_state.get("epochs", 15)
    batch_size = st.session_state.get("batch_size", 32)

    progress_bar = st.progress(0, text="Starting training...")

    class StreamlitProgress(__import__("tensorflow").keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            frac = (epoch + 1) / epochs
            progress_bar.progress(
                min(frac, 1.0),
                text=(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"loss: {logs.get('loss', 0):.4f} - "
                    f"accuracy: {logs.get('accuracy', 0):.4f} - "
                    f"val_loss: {logs.get('val_loss', 0):.4f} - "
                    f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
                )
            )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[StreamlitProgress()],
        verbose=0
    )

    # -------------------------------------------------------
    # Save Model
    # -------------------------------------------------------

    model.save(MODEL)

    emit("Model Saved Successfully!")

    # -------------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------------

    if status is not None:
        status.write("Evaluating model...")

    emit("Evaluating Model...")

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    st.metric("Test Loss", f"{loss:.4f}")
    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    # Metrics evaluation details
    predictions = (model.predict(X_test, verbose=0) > 0.8).astype(int).flatten()

    with st.expander("Classification Report & Confusion Matrix", expanded=False):
        st.write("**Classification Report**")
        st.code(classification_report(y_test, predictions, target_names=["Ham", "Spam"]))

        st.write("**Confusion Matrix**")
        st.code(str(confusion_matrix(y_test, predictions)))

    # -------------------------------------------------------
    # Plot Accuracy & Loss
    # -------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"])
    axes[0].plot(history.history["val_accuracy"])
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(["Train", "Validation"])

    axes[1].plot(history.history["loss"])
    axes[1].plot(history.history["val_loss"])
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(["Train", "Validation"])

    plt.tight_layout()

    st.pyplot(fig)

    return model, loss, accuracy


# ===========================================================
# Inference Helpers
# ===========================================================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Loads the trained model + tokenizer from disk.
    Cached so repeated predictions don't reload from disk each time.
    """

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


def predict_sms(message, model, tokenizer):
    """
    Takes a raw SMS message, cleans, tokenizes + pads it,
    runs it through the model, and returns a classification
    and its probability.
    """
    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = model.predict(padded, verbose=0)[0][0]
    prediction = "Spam" if probability > 0.8 else "Ham"

    return prediction, probability


def artifacts_available():
    return (
        os.path.exists(MODEL)
        and os.path.exists(TOKENIZER)
    )


# ===========================================================
# Streamlit Frontend
# ===========================================================

def main():

    st.set_page_config(
        page_title="SMS Spam Detector - Simple RNN",
        page_icon="💬",
        layout="centered"
    )

    st.title("💬 SMS Spam Detection")
    st.caption("Many-to-One Simple RNN, built with Keras + Streamlit")

    tab_predict, tab_train = st.tabs(["🔍 Classify Message", "⚙️ Train Model"])

    # -----------------------------------------------------
    # Tab: Predict
    # -----------------------------------------------------

    with tab_predict:

        st.subheader("Try the spam classifier")

        if not artifacts_available():
            st.warning(
                "No trained model found yet. Go to the "
                "**Train Model** tab first to train and save the model."
            )
        else:
            message = st.text_area(
                "Enter SMS Message",
                placeholder="e.g. Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."
            )

            if st.button("Predict spam/ham status", type="primary", use_container_width=True):
                if not message.strip():
                    st.error("Please enter a message first.")
                else:
                    with st.spinner("Loading model & predicting..."):
                        model, tokenizer = load_artifacts()
                        prediction, probability = predict_sms(message, model, tokenizer)

                    st.success("Done!")

                    # Show result nicely
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == "Spam":
                            st.error(f"Prediction: **{prediction}**")
                        else:
                            st.success(f"Prediction: **{prediction}**")
                    with col2:
                        st.metric("Spam Probability", f"{probability * 100:.2f}%")

    # -----------------------------------------------------
    # Tab: Train
    # -----------------------------------------------------

    with tab_train:

        st.subheader("Train / Retrain the Model")

        st.write(
            f"Dataset expected at: `{DATASET_PATH}` "
            "(must contain `v1` and `v2` columns where "
            "`v1` holds labels and `v2` holds message texts)."
        )

        if not os.path.exists(DATASET_PATH):
            st.error(f"Dataset file '{DATASET_PATH}' not found in the working directory.")
        else:
            st.success(f"Dataset file '{DATASET_PATH}' found.")

        col1, col2 = st.columns(2)

        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15, step=1)

        with col2:
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

        st.session_state["epochs"] = epochs
        st.session_state["batch_size"] = batch_size

        if artifacts_available():
            st.info("A trained model already exists. Training again will overwrite it.")

        if st.button(
            "Start Training",
            type="primary",
            use_container_width=True,
            disabled=not os.path.exists(DATASET_PATH)
        ):
            log_placeholder = st.empty()

            with st.status("Training in progress...", expanded=True) as status:
                try:
                    model, loss, accuracy = train_model(status=status, log=log_placeholder)
                    status.update(label="Training complete!", state="complete")

                    # Clear cached artifacts so the Predict tab picks up the new model
                    load_artifacts.clear()

                except Exception as e:
                    status.update(label="Training failed", state="error")
                    st.exception(e)


if __name__ == "__main__":
    main()