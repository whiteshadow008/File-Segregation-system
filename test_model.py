

import pandas as pd
import numpy as np
import joblib
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
except ImportError as e:
    print("Error: TensorFlow not installed or failed to load.")
    print(str(e))
    sys.exit(1)

# -----------------------------------------------------------
# 1. Data Loading and Preprocessing
# -----------------------------------------------------------

def load_data(file_path='keywords_dataset.csv'):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The dataset file is empty.")
        if df['keywords'].isna().any() or df['category'].isna().any():
            raise ValueError("Dataset contains missing values.")
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Data loading error: {e}")
        sys.exit(1)

# -----------------------------------------------------------
# 2. Traditional ML: TF-IDF + Naive Bayes
# -----------------------------------------------------------

def train_ml_model(X, y):
    try:
        print("--- Training Traditional ML Model ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        print("\n--- ML Model Performance ---")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # Save model and vectorizer
        joblib.dump(model, 'naive_bayes_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

        return model, vectorizer, X_test, y_test
    except Exception as e:
        print(f"Error during ML training: {e}")
        sys.exit(1)

# -----------------------------------------------------------
# 3. Deep Learning: Keras Model
# -----------------------------------------------------------

def train_dl_model(X, y, epochs=20):
    try:
        print("\n--- Training Deep Learning Model ---")
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        y_categorical = to_categorical(y_encoded, num_classes=num_classes)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )

        tokenizer = Tokenizer(oov_token='<unk>')
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        max_len = max(len(seq) for seq in X_train_seq)
        vocab_size = len(tokenizer.word_index) + 1

        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=max_len),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(X_train_pad, y_train, epochs=epochs, validation_split=0.1, verbose=1, callbacks=[callback])

        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
        print("\n--- DL Model Performance ---")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Save model and tokenizer
        model.save('dl_text_classifier.h5')
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)

        return model, tokenizer, encoder, max_len
    except Exception as e:
        print(f"Error during DL training: {e}")
        sys.exit(1)

# -----------------------------------------------------------
# 4. DL Prediction Function
# -----------------------------------------------------------

def predict_category(text, model, tokenizer, max_len, encoder):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(sequence, maxlen=max_len, padding='post')
        prediction = model.predict(padded_seq, verbose=0)
        class_index = np.argmax(prediction, axis=1)
        return encoder.inverse_transform(class_index)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# -----------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------

if __name__ == "__main__":
    df = load_data('keywords_dataset.csv')
    X, y = df['keywords'].astype(str), df['category']

    # Train ML model
    ml_model, vectorizer, X_test_ml, y_test_ml = train_ml_model(X, y)

    # Train DL model
    dl_model, tokenizer, encoder, max_len = train_dl_model(X, y, epochs=20)

    # Example DL prediction
    example_text = "loops conditionals functions oop"
    predicted_category = predict_category(example_text, dl_model, tokenizer, max_len, encoder)

    if predicted_category:
        print(f"\nExample text: '{example_text}'")
        print(f"Predicted category (DL model): {predicted_category}")
    else:
        print("Prediction failed.")
