# Spam Detection Project

A multi-model machine learning application for detecting spam emails/messages using Streamlit.

## Models Compared

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** ⭐ | **99.04%** | **99.64%** | 96.55% | **98.07%** |
| Random Forest | 98.78% | 98.94% | 96.21% | 97.55% |
| Naive Bayes | 98.43% | 97.89% | 95.86% | 96.86% |
| Logistic Regression | 98.43% | 99.64% | 94.14% | 96.81% |

**Best Model: SVM** with 99.04% accuracy

## Dataset

- **Total Samples:** 5,731
- **Used for Training:** 5,726 (after removing invalid labels)
- **Training Set:** 4,580 samples (80%)
- **Test Set:** 1,146 samples (20%)
- **Data Distribution:** 856 Ham, 290 Spam

## Project Structure

```
Spam_Detection_Project/
├── Data/
│   └── data.csv              # Dataset
├── saved_models/
│   ├── app.py               # Streamlit web application
│   ├── train_models.py      # Model training script
│   ├── evaluate_models.py   # Model evaluation & comparison
│   ├── requirements.txt     # Dependencies
│   └── *.pkl                # Trained models
└── README.md
```

## Installation

1. Clone the repository:
```bash
https://github.com/david1-max/spam-detection-system
cd Spam_Detection_Project
```

2. Install dependencies:
```bash
pip install -r saved_models/requirements.txt
```

## Usage

### Train Models
```bash
python saved_models/train_models.py
```

### Evaluate Models
```bash
python saved_models/evaluate_models.py
```

### Run Web Application
```bash
python -m streamlit run saved_models/app.py
```

Then open your browser at `http://localhost:8501`

## Features

- ✅ Multiple ML models (Naive Bayes, SVM, Logistic Regression, Random Forest)
- ✅ Interactive Streamlit web interface
- ✅ Real-time spam/ham classification
- ✅ Model selection dropdown
- ✅ High accuracy (99%+)
- ✅ Fast predictions

## Technologies Used

- Python 3.x
- Scikit-learn (Machine Learning)
- Pandas (Data Processing)
- Streamlit (Web Interface)
- TF-IDF Vectorization

## Performance Metrics

**SVM (Best Model):**
- Accuracy: 99.04%
- Precision: 99.64% (only 1 false positive)
- Recall: 96.55%
- False Positives: 1 (legitimate emails marked as spam)
- False Negatives: 10 (spam emails missed)

## Author

Divyanshu Yadav
Vaibhav Sharma
Raj Kanoi

## License

MIT
