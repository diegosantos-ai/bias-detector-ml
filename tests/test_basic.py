import pytest
import numpy as np
from src.ml.pipeline import SentenceEmbedder, BiasClassifier
from src.data.generator import generate_synthetic_data

def test_embedder_shape():
    """Embedder should return correct shape."""
    embedder = SentenceEmbedder()
    text = "Test sentence"
    emb = embedder.embed(text)
    assert emb.shape == (1, 384)
    
def test_data_generator():
    """Generator should return DataFrame with correct columns."""
    df = generate_synthetic_data(10)
    assert len(df) == 10
    assert "text" in df.columns
    assert "gender" in df.columns
    
def test_classifier_fit_predict():
    """Classifier should fit and predict correctly."""
    # 1. Gen data
    df = generate_synthetic_data(20)
    embedder = SentenceEmbedder()
    X = embedder.embed(df["text"].tolist())
    y = df[["gender", "age", "culture"]].values
    
    # 2. Fit
    clf = BiasClassifier()
    clf.fit(X, y)
    
    # 3. Predict
    probas = clf.predict_proba(X[:1])
    assert "gender" in probas
    assert isinstance(probas["gender"], np.ndarray)
