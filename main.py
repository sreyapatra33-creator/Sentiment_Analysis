
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.layers import GRU, LSTM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
CSV_FILE = "Twitter_Data.csv"      # Update path
TEXT_COLUMN = "selected_text"         # Update if needed
LABEL_COLUMN = "sentiment"    # Update if needed

VOCAB_SIZE = 20000
MAX_LENGTH = 50
EMBED_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10

# --------------- CLEANING ---------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------- LOAD DATA --------------
df = pd.read_csv(CSV_FILE)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df[LABEL_COLUMN])

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
test_seq = tokenizer.texts_to_sequences(X_test)

train_pad = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq, maxlen=MAX_LENGTH, padding="post"
)
test_pad = tf.keras.preprocessing.sequence.pad_sequences(
    test_seq, maxlen=MAX_LENGTH, padding="post"
)

num_classes = len(np.unique(y_train))

# --------------- MODELS -----------------
def build_simple_rnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        tf.keras.layers.SimpleRNN(64),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def build_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        tf.keras.layers.Bidirectional(LSTM(128)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def build_gru():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        tf.keras.layers.Bidirectional(GRU(128,dropout=0.3, recurrent_dropout=0.3)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

models = {
    "SimpleRNN": build_simple_rnn(),
    "LSTM": build_lstm(),
    "GRU": build_gru()
}

for name, model in models.items():
    print("="*60)
    print(name)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_pad,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    loss, acc = model.evaluate(test_pad, y_test, verbose=0)
    print("Test Accuracy:", acc)

    preds = model.predict(test_pad, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    print(confusion_matrix(y_test, pred_labels))
    print(classification_report(y_test, pred_labels))

    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train","Validation"])
    plt.tight_layout()
    plt.savefig(f"{name}_accuracy.png")
    plt.close()

    model.save(f"{name}.keras")

with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

with open("label_encoder.pkl","wb") as f:
    pickle.dump(encoder,f)

print("Training complete.")

def predict(text, model_file="LSTM.keras"):
    model=tf.keras.models.load_model(model_file)
    seq=tokenizer.texts_to_sequences([clean_text(text)])
    pad=tf.keras.preprocessing.sequence.pad_sequences(seq,maxlen=MAX_LENGTH,padding="post")
    pred=model.predict(pad,verbose=0)
    idx=int(np.argmax(pred))
    print("Prediction:",encoder.inverse_transform([idx])[0])
    print("Probabilities:",pred[0])

# Example:
# predict("I absolutely love this phone")
