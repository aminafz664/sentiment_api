from fastapi import FastAPI
from pydantic import BaseModel
from model import sentiment_model

app = FastAPI(title="Sentiment Analysis API")

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(data: TextInput):
    result = sentiment_model(data.text)[0]
    return {
        "sentiment": result["label"],
        "confidence": round(result["score"], 3)
    }