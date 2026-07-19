# Sentiment Analysis Using Deep Learning (RNN, LSTM & GRU)

## 📌 Project Overview

This project performs **Sentiment Analysis** on text data using three different Deep Learning models:

* Simple RNN (Recurrent Neural Network)
* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)

The application classifies text into one of the following sentiment categories:

* Negative
* Neutral
* Positive

The project also includes a **Streamlit web application** that allows users to enter text, choose a model, and predict the sentiment.

---

## 🚀 Features

* Text preprocessing using Tokenizer
* Sequence padding
* Label encoding and one-hot encoding
* Three Deep Learning models:

  * RNN
  * LSTM
  * GRU
* Model performance evaluation
* Interactive Streamlit web application
* Save and load trained models

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Pickle

---

## 📂 Project Structure

```
Sentiment-Analysis-App/
│
├── app.py
├── requirements.txt
├── README.md
│
├── RNN.keras
├── LSTM.keras
├── GRU.keras
│
├── tokenizer.pkl
├── label_encoder.pkl
│
└── dataset.csv
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Sentiment-Analysis-App.git
```

Move to the project directory:

```bash
cd Sentiment-Analysis-App
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 🧠 Models Used

* Simple RNN
* LSTM
* GRU

Each model is trained separately and saved as a `.keras` file.

---

## 📊 Evaluation

The models are evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

---

## 💾 Saved Files

* `RNN.keras`
* `LSTM.keras`
* `GRU.keras`
* `tokenizer.pkl`
* `label_encoder.pkl`

---

## 👩‍💻 Author

Developed as a Deep Learning Sentiment Analysis project using TensorFlow and Streamlit.
