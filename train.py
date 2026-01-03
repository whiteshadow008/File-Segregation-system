import os
import pickle
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# A dictionary to hold our loaded models
models = {}

# --- Helper functions for loading models and making predictions ---

def load_all_models():
    """Loads all models and components needed for the application."""
    try:
        # Load the ML model and vectorizer
        models['ml_vectorizer'] = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        models['ml_model'] = pickle.load(open('subject_classifier.pkl', 'rb'))
        print("Traditional ML model and vectorizer loaded successfully.")

        # Load the DL model and its components
        models['dl_tokenizer'] = pickle.load(open('tokenizer.pkl', 'rb'))
        models['dl_encoder'] = pickle.load(open('label_encoder.pkl', 'rb'))
        models['dl_model'] = tf.keras.models.load_model('subject_classifier.h5')
        print("Deep Learning model, tokenizer, and encoder loaded successfully.")

        # Get max sequence length from the DL tokenizer for padding
        max_len = 0
        for seq in models['dl_tokenizer'].texts_to_sequences(['placeholder text']):
            if len(seq) > max_len:
                max_len = len(seq)
        models['dl_max_len'] = max_len
        
    except FileNotFoundError as e:
        print(f"Error loading model file: {e}. Please ensure all .pkl and .h5 files are in the same directory.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading models: {e}")
        exit()

def predict_ml(text):
    """Makes a prediction using the traditional ML model."""
    vectorizer = models['ml_vectorizer']
    model = models['ml_model']
    
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

def predict_dl(text):
    """Makes a prediction using the deep learning model."""
    tokenizer = models['dl_tokenizer']
    encoder = models['dl_encoder']
    model = models['dl_model']
    max_len = models['dl_max_len']
    
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Make a prediction
    prediction = model.predict(padded_sequence)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_category = encoder.inverse_transform([predicted_class_index])[0]
    return predicted_category

# --- Flask App Setup ---
app = Flask(__name__)

# Load all models when the app starts
load_all_models()

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making a prediction."""
    text = request.form.get('text_input')
    model_choice = request.form.get('model_choice')
    
    if not text:
        return jsonify({'error': 'Please enter some text.'}), 400
    
    if model_choice == 'ml':
        prediction = predict_ml(text)
    elif model_choice == 'dl':
        prediction = predict_dl(text)
    else:
        return jsonify({'error': 'Invalid model choice.'}), 400
        
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    # Use 0.0.0.0 to make the app accessible from other machines
    # In a production environment, use a proper web server like Gunicorn or uWSGI
    app.run(debug=True, host='0.0.0.0', port=5000)
