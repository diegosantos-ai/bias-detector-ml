import pytest
import numpy as np
from src.ml.pipeline import SentenceEmbedder, BiasClassifier
from src.data.generator import generate_synthetic_data

def test_embedder_shape():
    """O Embedder deve retornar o formato correto."""
    embedder = SentenceEmbedder()
    text = "Frase de teste"
    emb = embedder.embed(text)
    assert emb.shape == (1, 384)
    
def test_data_generator():
    """O gerador deve retornar um DataFrame com as colunas corretas."""
    df = generate_synthetic_data(10)
    assert len(df) == 10
    assert "text" in df.columns
    assert "gender" in df.columns
    
def test_classifier_fit_predict():
    """O classificador deve treinar e prever corretamente."""
    # 1. Gerar dados
    df = generate_synthetic_data(20)
    embedder = SentenceEmbedder()
    # Traduzido para consistência, mas o conteúdo do texto não afeta a lógica do teste
    X = embedder.embed(df["text"].tolist())
    y = df[["gender", "age", "culture"]].values
    
    # 2. Treinar
    clf = BiasClassifier()
    clf.fit(X, y)
    
    # 3. Prever
    probas = clf.predict_proba(X[:1])
    assert "gender" in probas
    assert isinstance(probas["gender"], np.ndarray)
