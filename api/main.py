from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords
from app.label_list import label_list

model = joblib.load('models/svm_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def clean_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    return ' '.join([w for w in text.split() if w not in stopwords.words('english')])

app = FastAPI()

class Comment(BaseModel):
    text: str

@app.post("/predict")
def predict_toxicity(comment: Comment):
    cleaned = clean_text(comment.text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    predicted_labels = [label_list[i] for i, v in enumerate(pred[0]) if v == 1]
    return {"labels": predicted_labels}
