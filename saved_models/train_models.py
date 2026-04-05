import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
# Note: encoding='latin-1' is often required for this specific dataset
df = pd.read_csv('data/data.csv', encoding='latin-1')
df = df[['text', 'spam']]
df.columns = ['message', 'label']
# Convert label: '1' = spam, '0' = ham
df['label'] = pd.to_numeric(df['label'], errors='coerce')
# Remove rows with missing or invalid label values
df = df.dropna()
df['message'] = df['message'].astype(str)

# 2. Preprocessing & Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X = tfidf.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the 4 Models
models = {
    "Naive_Bayes": MultinomialNB(),
    "SVM": SVC(probability=True),
    "Logistic_Regression": LogisticRegression(),
    "Random_Forest": RandomForestClassifier(n_estimators=100)
}

# 4. Train and Save
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    with open(f'saved_models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save the vectorizer (the app needs this to convert new input)
with open('saved_models/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nSuccess! All models and the vectorizer are saved in /saved_models")