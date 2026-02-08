"""
Aplicação FastAPI para Detecção de Viés.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.ml.pipeline import SentenceEmbedder, BiasClassifier
from src.data.generator import generate_synthetic_data # Para teste rápido
import pandas as pd

app = FastAPI(title="API Detector de Viés em RH")

# Carregar modelo (Carregamento simulado por enquanto, deve carregar de arquivo/MLflow)
embedder = SentenceEmbedder()
# Treinar um modelo fictício na inicialização para fins de demonstração
print("Treinando modelo de demonstração...")
df = generate_synthetic_data(100)
X = embedder.embed(df["text"].tolist())
y = df[["gender", "age", "culture"]].values
classifier = BiasClassifier()
classifier.fit(X, y)
print("Modelo pronto!")

class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    text: str
    scores: dict[str, float]
    flags: list[str]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    # 1. Incorporar (Embed)
    embedding = embedder.embed(request.text)
    
    # 2. Prever
    probas = classifier.predict_proba(embedding)
    
    # 3. Formatar
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
