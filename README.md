
# Toxic Comment Classification (SVM-based Web App)
The project is machine learning web application that classifies user input text into multiple toxic comment categories using an **SVM (Support Vector Machines) model**.

## Features
* **Multi-label classification** within 6 toxic categories (toxic, severe_toxic, obscene, threat, insult, identity_hate)
* Machine learning model trained on **Jigsaw Toxic Comment Classification dataset**
* Frontend: **Streamlit** (Python)
* Backend: **FastAPI**
* Saving the trained model in a **pickle format** for deployment (*.pkl*).




## Intsalling Locally

#### Before of all, install Python!

### 1. Clone the repository

```git
  git clone https://github.com/SkalapEnder/ML_Final2
```

### 2. Install dependencies

```
  pip install -r requirements.txt
```

### 3. Train the model

```
  python app/train_model.py
```

### 4. Run the API locally

```
  uvicorn api.main:app --reload
```

### 5. Run the application locally

```
  streamlit run frontend/app.py
```


## Project Structure
```
.
├── api/
│   ├── main.py                 # FastAPI backend
├── app/
│   ├── label_list.py           # List of labels
│   ├── train_model.py          # Training model script
├── data/
│   ├── train_clean.csv         # Datasets
│   ├── test_clean.csv          
│   ├── test_labels.csv        
├── frontend/
│   ├── app.py                  # Streamlit application
├── models/
│   ├── svm_model.pkl           # ML model (Support Vector Machines)
│   ├── tfidf_vectorizer.pkl    # TF-IDF Vectorizer
├── requirements.txt            # Python dependencies
└── README.md
```

## Examples
There are several input examples to check model:

### Example 1
Input:
```
Come out, dirty animal!
```

Output:
```
Predicted Labels: toxic, insult
```


### Example 2
Input:
```
I’ll find you and kill you, stinking nerd.
```

Output:
```
Predicted Labels: toxic, threat, insult
```

### Example 3
Input:
```
You're such a nasty piece of trash.
```

Output:
```
Predicted Labels: toxic, insult
```




## Author

Alisher Berik, IT-2308, Astana IT University
