"""
Script de treinamento com rastreamento MLflow.
"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data.generator import generate_synthetic_data
from src.ml.pipeline import SentenceEmbedder, BiasClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(samples=500, test_size=0.2):
    mlflow.set_experiment("bias-detector-ml")
    
    # Tags de Experimento
    mlflow.set_tag("context", "hr-recruitment")
    mlflow.set_tag("model_type", "logistic-regression")
    mlflow.set_tag("dataset", "synthetic-v1")
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("developer", "diego-santos")
    
    # Verificar se já existe uma run ativa para evitar erro de aninhamento
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        # 1. Gerar Dados
        logger.info(f"Gerando {samples} amostras...")
        df = generate_synthetic_data(samples)
        X_text = df["text"].tolist()
        y = df[["gender", "age", "culture"]].values
        
        # 2. Embeddings
        logger.info("Gerando embeddings...")
        embedder = SentenceEmbedder()
        X = embedder.embed(X_text)
        
        # 3. Divisão
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 4. Treinamento
        logger.info("Treinando classificador...")
        clf = BiasClassifier()
        clf.fit(X_train, y_train)
        
        # 5. Avaliação
        logger.info("Avaliando...")
        y_pred_proba = clf.predict_proba(X_test)
        # Converter dict de probabilidades para matriz binária para relatório
        # Avaliação simplificada para demonstração
        
        mlflow.log_param("samples", samples)
        mlflow.log_param("model", "all-MiniLM-L6-v2")
        
        # Salvar modelo e registrar
        mlflow.sklearn.log_model(
            clf.clf, 
            "model",
            registered_model_name="bias-detector-ml"
        )
        logger.info("Modelo salvo e registrado no MLflow como 'bias-detector-ml'.")

if __name__ == "__main__":
    train()
