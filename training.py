import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import pandas as pd

# Parameters
VOCAB_SIZE = 10000
MAX_LEN = 250
EMBEDDING_DIM = 16
MODEL_PATH = 'sentiment_analysis_model.keras'  # Updated extension for TensorFlow 2.x
TOKENIZER_PATH = 'tokenizer.pickle'

# Load dataset
file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
df_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract texts and labels
texts = df_shuffled.iloc[:, -1].astype(str).tolist()
labels = df_shuffled.iloc[:, 0].apply(lambda x: 0 if x == 0 else 1 if x == 2 else 2).to_numpy()

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Save tokenizer
with open(TOKENIZER_PATH, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Split data
train_size = int(0.8 * len(padded_sequences))  # 80% train, 20% test
train_data, test_data = padded_sequences[:train_size], padded_sequences[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Load or train model
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
else:
    print("Training a new model...")
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save model
    model.save(MODEL_PATH)

# Evaluate model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Sentiment prediction function
def encode_text(text):
    tokens = tokenizer.texts_to_sequences([text])
    return pad_sequences(tokens, maxlen=MAX_LEN, padding='post', truncating='post')

while True:
    user_input = input("Enter a sentence for sentiment analysis (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    encoded_input = encode_text(user_input)
    prediction = np.argmax(model.predict(encoded_input))
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"Sentiment: {sentiment_labels[prediction]}")
