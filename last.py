import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# 1. Load CSV Data
# -------------------------------
def load_data(file_path='keywords_dataset.csv'):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty.")
        if df['keywords'].isna().any() or df['category'].isna().any():
            raise ValueError("Missing values detected.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# -------------------------------
# 2. Train RNN Model
# -------------------------------
def train_rnn_model(X, y, epochs=20):
    try:
        print("\n--- Training RNN (Bidirectional LSTM) Model ---")
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        y_categorical = to_categorical(y_encoded, num_classes=num_classes)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Tokenizer
        tokenizer = Tokenizer(oov_token='<unk>')
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        # Padding
        max_len = max(len(seq) for seq in X_train_seq)
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
        vocab_size = len(tokenizer.word_index) + 1

        # Build RNN Model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # Early stopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train model
        model.fit(X_train_pad, y_train, epochs=epochs, validation_split=0.1, batch_size=32, callbacks=[callback])

        # Evaluate
        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
        print(f"\n--- RNN Model Test Accuracy: {accuracy:.4f} ---")

        # Save model, tokenizer, encoder
        model.save('rnn_text_classifier.h5')
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        with open('max_len.pkl', 'wb') as f:
            pickle.dump(max_len, f)


        return model, tokenizer, encoder, max_len

    except Exception as e:
        print(f"Error training RNN model: {e}")
        sys.exit(1)

# -------------------------------
# 3. Predict Function
# -------------------------------
def predict_category(text, model, tokenizer, max_len, encoder):
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        pred = model.predict(padded, verbose=0)
        class_index = np.argmax(pred, axis=1)
        return encoder.inverse_transform(class_index)[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# -------------------------------
# 4. Main Execution
# -------------------------------
if __name__ == "__main__":
    df = load_data('keywords_dataset.csv')
    X, y = df['keywords'].astype(str), df['category']

    # Train RNN Model
    rnn_model, tokenizer, encoder, max_len = train_rnn_model(X, y, epochs=20)

    # Example prediction
    example_text = "loops conditionals functions oop"
    predicted_category = predict_category(example_text, rnn_model, tokenizer, max_len, encoder)
    print(f"\nExample Text: '{example_text}'")
    print(f"Predicted Category (RNN): {predicted_category}")
