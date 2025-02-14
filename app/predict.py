import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import gdown
import os

VOCAB_SIZE = 10000
MAX_LEN = 250
# Define Google Drive file IDs (replace with your actual file IDs)
MODEL_DRIVE_ID = "1NPeTOQKzjXYO5xiqJrczrXU1JAq2yjid"
TOKENIZER_DRIVE_ID = "1c_LQA1wDXq9Z3RmYUF3WT6e8fh8JUqqD"

# Define local paths
MODEL_PATH = "../sentiment_analysis_model.keras"
TOKENIZER_PATH = "../tokenizer.pickle"



# Function to download files if they don’t exist
def download_file(drive_id, output_path):
    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={drive_id}", output_path, quiet=False)

# Download model and tokenizer if not available locally
download_file(MODEL_DRIVE_ID, MODEL_PATH)
download_file(TOKENIZER_DRIVE_ID, TOKENIZER_PATH)

# Load the saved model
model = load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

print(len(tokenizer.word_index))  # Check vocabulary size



def encode_texts(text_list):
    encoded_texts = []
    for text in text_list:
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [
            tokenizer.word_index[word] if word in tokenizer.word_index and tokenizer.word_index[word] < VOCAB_SIZE else 0
            for word in tokens
        ]
        encoded_texts.append(tokens)
    return pad_sequences(encoded_texts, maxlen=MAX_LEN, padding='post', value=0)



def predict_sentiments(text_list):
    encoded_inputs = encode_texts(text_list)
    print(encoded_inputs)  # Check the encoded inputs
    print("Max token index in encoded input:", np.max(encoded_inputs))
    predictions = np.argmax(model.predict(encoded_inputs), axis=-1)
    
    sentiments = ["Negative" if p == 0 else "Neutral" if p == 1 else "Positive" for p in predictions]
    return sentiments




