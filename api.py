from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = joblib.load('spam_model.pkl')
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
    return {"prediction": int(y_pred)}
