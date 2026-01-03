import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================
# 1. Training Data (expandable)
# =============================
training_data = [
    # AI
    ("neural networks deep learning artificial intelligence models supervised learning unsupervised learning", "AI"),
    ("cnn rnn lstm transformers attention mechanism text summarization nlp", "AI"),

    # ML
    ("decision trees clustering regression svm ensemble boosting bagging", "ML"),

    # DBMS
    ("sql normalization relational databases indexing joins queries transactions", "DBMS"),

    # OS
    ("scheduling deadlock threads memory management process states", "OS"),

    # CN
    ("ip addressing tcp udp protocols osi layers network routing switching", "CN"),

    # Programming
    ("python programming loops conditionals functions oop exception handling", "Programming"),
    ("c java oop polymorphism inheritance encapsulation abstraction interfaces", "Programming"),

    # DSA
    ("arrays linked lists stacks queues hashmaps trees graphs algorithms", "DSA"),

    # Cloud
    ("cloud computing virtualization iaas paas saas aws azure gcp", "Cloud"),

    # Data Visualization (DV)
    ("data visualization matplotlib seaborn dashboards charts plots graphs", "DV"),
    ("tableau powerbi dashboards data presentation bar chart line chart scatter plot", "DV"),
]

texts, labels = zip(*training_data)

# Unique subjects
subjects = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(subjects)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

y = np.array([label_to_index[label] for label in labels])

# =============================
# 2. Tokenization
# =============================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=20)

# =============================
# 3. Model
# =============================
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=20),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dense(len(subjects), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# =============================
# 4. Training
# =============================
model.fit(X, y, epochs=30, verbose=1)

# =============================
# 5. Save Model & Tokenizer
# =============================
model.save("subject_classifier.h5")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model training complete. Subjects trained:", subjects)
