import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('data/train_clean.csv')

# Labels
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df = df.dropna(subset=['comment_text'])

# Text cleaning
def clean_text(text):
    text = re.sub(r'\W', ' ', text)         
    text = text.lower()                    
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

print("Cleaning data...")
df['clean_comment'] = df['comment_text'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_comment'])
y = df[label_cols].values

# Save label list
# with open('app/label_list.py', 'w') as f:
#     f.write(f"label_list = {label_cols}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = OneVsRestClassifier(LinearSVC())
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_cols))

# Save model and vectorizer
joblib.dump(model, 'models/svm_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
