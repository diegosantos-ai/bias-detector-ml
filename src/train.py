"""
Training script with MLflow tracking.
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
    
    with mlflow.start_run():
        # 1. Generate Data
        logger.info(f"Generating {samples} samples...")
        df = generate_synthetic_data(samples)
        X_text = df["text"].tolist()
        y = df[["gender", "age", "culture"]].values
        
        # 2. Embeddings
        logger.info("Generating embeddings...")
        embedder = SentenceEmbedder()
        X = embedder.embed(X_text)
        
        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 4. Train
        logger.info("Training classifier...")
        clf = BiasClassifier()
        clf.fit(X_train, y_train)
        
        # 5. Evaluate
        logger.info("Evaluating...")
        y_pred_proba = clf.predict_proba(X_test)
        # Convert dict of probas to binary matrix for report
        # Simplified evaluation for demo
        
        mlflow.log_param("samples", samples)
        mlflow.log_param("model", "all-MiniLM-L6-v2")
        
        # Save model
        mlflow.sklearn.log_model(clf.clf, "model")
        logger.info("Model saved to MLflow.")

if __name__ == "__main__":
    train()
