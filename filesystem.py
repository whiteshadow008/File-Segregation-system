import os
import shutil
import docx
from pdfminer.high_level import extract_text as extract_pdf_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# ---------- Sample Training Data ----------
training_data = [
    ('integration limits calculus derivatives', 'Mathematics'),
    ('programming oop object oriented python java', 'Computer_Science'),
    ('kinematics thermodynamics gravity motion', 'Physics'),
    ('poetry grammar prose shakespeare novel', 'English'),
    ('linked list stack queue algorithm', 'Computer_Science'),
    ('optics lens wave refraction', 'Physics'),
    ('algebra equation matrix geometry', 'Mathematics'),
    ('comprehension essay writing literature', 'English')
]
texts, labels = zip(*training_data)

# ---------- Model Training ----------
model = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])
model.fit(texts, labels)

# ---------- File Text Extractor ----------
def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf':
            return extract_pdf_text(filepath)
        elif ext == '.docx':
            doc = docx.Document(filepath)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    except:
        return ""
    return ""

# ---------- Main Segregation Logic ----------
def segregate_files(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder does not exist: {input_folder}")
        return

    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(f"\nFound {len(files)} file(s) in input folder: '{input_folder}'")

    os.makedirs(output_folder, exist_ok=True)

    for file in files:
        filepath = os.path.join(input_folder, file)
        text = extract_text(filepath)
        if text.strip():
            predicted_label = model.predict([text])[0]
            dest_folder = os.path.join(output_folder, predicted_label)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(filepath, os.path.join(dest_folder, file))
            print(f"Moved '{file}' → {predicted_label}")
        else:
            print(f"Skipped '{file}': empty or unreadable")

    # Count and display results
    print(f"\nFiles organized into subject folders under: '{output_folder}'\n")
    total_out = 0
    for root, dirs, files in os.walk(output_folder):
        for name in files:
            total_out += 1
        for d in dirs:
            folder_path = os.path.join(root, d)
            count = len(os.listdir(folder_path))
            print(f"  → {d}: {count} file(s)")

    print(f"\nTotal organized files: {total_out}\n")

# ---------- Run the System ----------
if __name__ == '__main__':
    print("\n=== File Segregation System ===")
    input_path = r"C:\Users\Dhanush\OneDrive\Desktop\Input Folder".strip()

    output_path = r"C:\Users\Dhanush\OneDrive\Desktop\Output Folder".strip()

    print("\n--- File Segregation Started ---")
    segregate_files(input_path, output_path)
    print("--- File Segregation Completed ---\n")
