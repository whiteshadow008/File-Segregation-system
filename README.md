# 📁 Intelligent File Segregation System using Machine Learning

This project implements an intelligent file segregation system that classifies and organizes files using machine learning techniques rather than simple file-extension rules.
It learns from keyword datasets and predicts the appropriate category for files, providing more accurate and flexible organization.

## 🧠 Key Features

- ML-based file classification
- Keyword-driven training dataset
- Web interface using Flask
- Model training and testing pipeline
- Automated file organization logic

## 🛠 Tech Stack

- Python
- Machine Learning (custom training pipeline)
- Flask (web interface)
- HTML (templates)
- CSV-based dataset

## 📂 Project Structure
File-Segrication-System/

│

├── filesystem.py          # Core file handling & segregation logic

├── main.py                # Application entry point

├── new.py / last.py       # Supporting execution logic

├── train.py               # Model training script

├── train_model.py         # ML model builder

├── test_model.py          # Model testing & validation

├── keywords_dataset.csv   # Training dataset

│

├── templates/

│   ├── index.html         # UI page

│   └── result.html        # Result display

│

├── README.md

├── .gitignore

└── venv/                  # (Local only – see note below)


## ▶️ How to Run
### Create virtual environment
python -m venv venv

source venv/bin/activate   # Windows: venv\Scripts\activate

### Install dependencies (if requirements.txt is added later)
pip install -r requirements.txt

### Run application
python main.py

## 🚀 Future Improvements

- Add requirements.txt
- Remove venv/ from repository
- Improve model accuracy with larger datasets
- Add drag-and-drop UI for files
- Logging and error handling



