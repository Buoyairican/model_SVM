from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = joblib.load('SVM.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # optional if you're using text input

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
@app.post("/predict")
def predict(data: Message):
    X = vectorizer.transform([data.text])
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]  # returns [prob_ham, prob_spam]

    return {
        "prediction": int(y_pred),
        "probabilities": {
            "ham": round(float(y_proba[0]), 4),
            "spam": round(float(y_proba[1]), 4)
        }
    }
