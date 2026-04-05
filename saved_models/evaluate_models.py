import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/data.csv', encoding='latin-1')
df = df[['text', 'spam']]
df.columns = ['message', 'label']
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna()
df['message'] = df['message'].astype(str)

# 2. Prepare data the same way as training
tfidf = pickle.load(open('saved_models/vectorizer.pkl', 'rb'))
X = tfidf.transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Load all models
models = {
    "Naive_Bayes": pickle.load(open('saved_models/Naive_Bayes.pkl', 'rb')),
    "SVM": pickle.load(open('saved_models/SVM.pkl', 'rb')),
    "Logistic_Regression": pickle.load(open('saved_models/Logistic_Regression.pkl', 'rb')),
    "Random_Forest": pickle.load(open('saved_models/Random_Forest.pkl', 'rb'))
}

# 4. Evaluate all models
print("\n" + "="*80)
print("MODEL EVALUATION & COMPARISON")
print("="*80)

results = {}

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# 5. Comparison Summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame(results).T
print("\n", comparison_df.to_string())

# 6. Rankings
print("\n" + "="*80)
print("MODEL RANKINGS")
print("="*80)

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    print(f"\n{metric} Rankings:")
    ranked = comparison_df[metric].sort_values(ascending=False)
    for rank, (model, score) in enumerate(ranked.items(), 1):
        print(f"  {rank}. {model}: {score:.4f} ({score*100:.2f}%)")

# 7. Best Model
best_accuracy = comparison_df['Accuracy'].idxmax()
best_f1 = comparison_df['F1-Score'].idxmax()

print("\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)
print(f"\nBest Model (by Accuracy): {best_accuracy} - {comparison_df.loc[best_accuracy, 'Accuracy']:.4f}")
print(f"Best Model (by F1-Score): {best_f1} - {comparison_df.loc[best_f1, 'F1-Score']:.4f}")
print(f"\nTest Set Size: {len(y_test)} samples")
print(f"Training Set Size: {len(y_train)} samples")
print(f"Spam vs Ham in Test Set: {int(y_test.sum())} spam, {len(y_test) - int(y_test.sum())} ham")
