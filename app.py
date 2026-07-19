import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# Constants
# -----------------------
MAX_LENGTH = 100

# -----------------------
# Load Tokenizer
# -----------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------
# Load Label Encoder
# -----------------------
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------
# Streamlit App
# -----------------------
st.title("Sentiment Analysis using Deep Learning")
st.write("Predict sentiment using RNN, LSTM, or GRU.")

# Select model
model_name = st.selectbox(
    "Choose a Model",
    ["RNN", "LSTM", "GRU"]
)

# Load selected model
model = tf.keras.models.load_model(f"{model_name}.keras")

# User input
text = st.text_area("Enter your text:")

# Predict
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([text])

        # Pad sequence
        pad = pad_sequences(
            seq,
            maxlen=MAX_LENGTH,
            padding="post"
        )

        # Prediction
        prediction = model.predict(pad, verbose=0)

        # Highest probability class
        predicted_index = np.argmax(prediction)

        # Convert index to sentiment label
        sentiment = label_encoder.inverse_transform([predicted_index])[0]

        st.success(f"Predicted Sentiment: {sentiment}")

        st.subheader("Prediction Probabilities")
        st.write(prediction[0])
        