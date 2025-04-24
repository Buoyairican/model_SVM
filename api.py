from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import string
import re
import string 
# Load model and vectorizer
model = joblib.load('SVM.pkl')
vectorizer = joblib.load('vectorizer.pkl')
stopwordsSet = joblib.load('stopwords.pkl')
stemmer = joblib.load('stemmer.pkl')  # Usually a PorterStemmer or similar object

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocessing function
def preprocess_email(email_text):
    # 1. Convert to lowercase
    text = email_text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 4. Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # 5. Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 6. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 7. Remove digits
    text = re.sub(r'\d+', ' ', text)

    # 8. Tokenization, stopword removal, stemming
    tokens = []
    for tok in text.split():
        if tok not in stopwordsSet and len(tok) > 1:
            tokens.append(stemmer.stem(tok))

    # 9. Rejoin tokens
    return ' '.join(tokens)

# Input schema
class Message(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(data: Message):
    email_clean = preprocess_email(data.text)
    email_vec = vectorizer.transform([email_clean])
    y_pred = model.predict(email_vec)[0]
    y_proba = model.predict_proba(email_vec)[0]

    return {
        "prediction": int(y_pred),
        "probabilities": {
            "ham": round(float(y_proba[0]), 4),
            "spam": round(float(y_proba[1]), 4)
        }
    }
