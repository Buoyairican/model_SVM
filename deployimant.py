from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # optional if you're using text input

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Message):
    X = vectorizer.transform([data.text])
    y_pred = model.predict(X)[0]
    return {"prediction": int(y_pred)}