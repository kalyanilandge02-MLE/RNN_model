import streamlit as st
import numpy as np
import pickle
import os
import requests
import zipfile
from azure.storage.blob import BlobServiceClient
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = "<your_connection_string>"
BLOB_CONTAINER_NAME = "<your_container_name>"
MODEL_BLOB_NAME = "rnn_saved_model"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "rnn_saved_model")
MODEL_URL = "https://kalyanimlmodels.blob.core.windows.net/mlmodels/rnn_saved_model.zip?sp=r&st=2026-01-15T18:13:09Z&se=2026-01-16T02:28:09Z&spr=https&sv=2024-11-04&sr=b&sig=zJFOMU4tgiWrRW64Ck1Q8wlnHxARnMa8GIR6DZ93zfY%3D"

# Azure Blob Storage configuration for tokenizer
TOKENIZER_BLOB_NAME = "tokenizer.pickle"
TOKENIZER_PATH = os.path.join(MODEL_DIR, TOKENIZER_BLOB_NAME)
TOKENIZER_URL = "https://kalyanimlmodels.blob.core.windows.net/mlmodels/tokenizer.pickle?sp=r&st=2026-01-15T18:48:07Z&se=2026-01-16T03:03:07Z&spr=https&sv=2024-11-04&sr=b&sig=X9PPygbB6TgCB9sBzHUvvaTuc8WojP2gFVrDyznr954%3D"

# Ensure the local model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model from Azure Blob Storage
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists — skipping download")
        return

    zip_path = os.path.join(MODEL_DIR, "model.zip")

    r = requests.get(MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    print("✅ SavedModel extracted")

# Download the tokenizer from Azure Blob Storage
def download_tokenizer():
    if os.path.exists(TOKENIZER_PATH):
        print("✅ Tokenizer already exists — skipping download")
        return

    r = requests.get(TOKENIZER_URL, stream=True, timeout=120)
    r.raise_for_status()

    with open(TOKENIZER_PATH, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print("✅ Tokenizer downloaded successfully")

# Lazy model loader (loads only once)
model = None

def get_model():
    global model

    if model is None:
        print("🧠 Loading model into memory...")
        download_model()

        model = tf.keras.models.load_model(MODEL_PATH)

        print("✅ Model loaded successfully")

    return model

# Ensure tokenizer is downloaded
download_tokenizer()

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Replace the existing model loading logic with the new one
model = get_model()

# Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequences length match
    token_list = pad_sequences([token_list],maxlen = max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose =0)
    predicted_word_index = np.argmax(predicted,axis = 1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
# Streamlit app
st.title("Next word prediction with LSTM And Early Stopping ")
input_text = st.text_input("Enter the Sequence of words","To be or not to")
if st.button("predict next word"):
    max_sequence_len = model.input_shape[1] + 1 # Retrive the max sequence length from th model")
    next_word = predict_next_word(model, tokenizer,input_text,max_sequence_len)
    st.write(f'Next word:{next_word}')




