"""
Componentes do Pipeline de ML: Embedder e Classificador.
"""
from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

class SentenceEmbedder:
    """Wrapper para SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)


class BiasClassifier:
    """Classificador Multi-rótulo."""
    
    def __init__(self, labels: list[str] = None):
        self.labels = labels or ["gender", "age", "culture"]
        self.clf = OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", random_state=42)
        )
        self.mlb = MultiLabelBinarizer(classes=self.labels)
        self.is_fitted = False
        
    def fit(self, X, y):
        """Treinar modelo. y é uma lista de listas de rótulos."""
        y_binary = np.array(y)  # Assumindo que y já é uma matriz binária do gerador
        self.clf.fit(X, y_binary)
        self.is_fitted = True
        return self
        
    def predict_proba(self, X):
        """Retornar dicionário de probabilidades."""
        probs = self.clf.predict_proba(X)
        return {
            label: probs[:, i] 
            for i, label in enumerate(self.labels)
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
