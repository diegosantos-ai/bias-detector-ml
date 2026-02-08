"""
FastAPI Application for Bias Detection.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.ml.pipeline import SentenceEmbedder, BiasClassifier
from src.data.generator import generate_synthetic_data # For quick testing
import pandas as pd

app = FastAPI(title="HR Bias Detector API")

# Load model (Mock loading for now, should load from file/MLflow)
embedder = SentenceEmbedder()
# Train a dummy model on startup for demo purposes
print("Training demo model...")
df = generate_synthetic_data(100)
X = embedder.embed(df["text"].tolist())
y = df[["gender", "age", "culture"]].values
classifier = BiasClassifier()
classifier.fit(X, y)
print("Model ready!")

class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    text: str
    scores: dict[str, float]
    flags: list[str]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    # 1. Embed
    embedding = embedder.embed(request.text)
    
    # 2. Predict
    probas = classifier.predict_proba(embedding)
    
    # 3. Format
    scores = {k: float(v[0]) for k, v in probas.items()}
    flags = [k for k, v in scores.items() if v > 0.5]
    
    return {
        "text": request.text,
        "scores": scores,
        "flags": flags
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
