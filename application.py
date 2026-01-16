from flask import Flask, request, jsonify, render_template
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import zipfile
import logging
from requests.exceptions import HTTPError

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Model and Tokenizer Paths
# --------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "rnn_saved_model")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")

# Model URL and Tokenizer URL
MODEL_URL = "https://kalyanimlmodels.blob.core.windows.net/mlmodels/rnn_saved_model.zip?sp=r&st=2026-01-16T04:14:13Z&se=2026-01-30T12:29:13Z&spr=https&sv=2024-11-04&sr=b&sig=6JMr7HLHeF0Dr8Ys3CxR8QaGibk6crWDXo81tTRVJBA%3D"
TOKENIZER_URL = "https://kalyanimlmodels.blob.core.windows.net/mlmodels/tokenizer.pickle?sp=r&st=2026-01-16T04:14:59Z&se=2026-01-30T12:29:59Z&spr=https&sv=2024-11-04&sr=b&sig=k8AbMffWhSuabtndeM2D55Inw0hzRzp6BMo5c3vwnzQ%3D"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# Download Model and Tokenizer
# --------------------------------------------------
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists — skipping download")
        return

    zip_path = os.path.join(MODEL_DIR, "model.zip")

    try:
        r = requests.get(MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        print("✅ SavedModel extracted")

    except HTTPError as http_err:
        logging.error(f"HTTP error occurred while downloading the model: {http_err}")
        raise
    except Exception as err:
        logging.error(f"An error occurred while downloading the model: {err}")
        raise

def download_tokenizer():
    r = requests.get(TOKENIZER_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(TOKENIZER_PATH, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

# Download if not already present
if not os.path.exists(MODEL_PATH):
    download_model()
if not os.path.exists(TOKENIZER_PATH):
    download_tokenizer()

# --------------------------------------------------
# Lazy Model Loader
# --------------------------------------------------
model = None
def get_model():
    global model
    if model is None:
        print("🧠 Loading model into memory...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    return model

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    if not text:
        return render_template("index.html", prediction="No input text provided")

    mdl = get_model()
    max_sequence_len = mdl.input_shape[1] + 1
    next_word = predict_next_word(mdl, tokenizer, text, max_sequence_len)
    return render_template("index.html", prediction=next_word)

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------------------
# Local Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)




