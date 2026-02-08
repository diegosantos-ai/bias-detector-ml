# HR Bias Detector ML

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)](https://mlflow.org/)

Machine Learning model to detect implicit biases (gender, age, culture) in job descriptions. Built with Sentence-Transformers and tracked with MLflow.

## 🚀 Key Features

- **Multi-label Classification**: Detects multiple types of bias simultaneously.
- **Explainability**: Categorizes bias type (e.g., "Ageism", "Gender Bias").
- **MLflow Integration**: Full experiment tracking and model registry.
- **REST API**: FastAPI endpoint for real-time analysis.

## 🛠️ Stack

- **ML**: `scikit-learn`, `sentence-transformers`
- **Ops**: `mlflow`, `docker`
- **API**: `fastapi`

## 📦 Installation

```bash
# Clone
git clone https://github.com/yourusername/bias-detector-ml.git
cd bias-detector-ml

# Env
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Deps
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

1. **Start MLflow (Optional)**
   ```bash
   mlflow ui --port 5001
   ```

2. **Train Model**
   ```bash
   # Generates synthetic data and trains the model
   python -m src.train
   ```
   *Note: First run will download the embedding model (80MB).*

3. **Run API**
   ```bash
   uvicorn src.api.app:app --reload
   ```
   Access docs at: http://localhost:8000/docs

## 🧪 Testing

```bash
# Run unit tests
pytest tests/
```

## 📂 Project Structure

```
bias-detector-ml/
├── src/
│   ├── api/            # API endpoints
│   ├── data/           # Data generation
│   ├── ml/             # ML pipeline
│   └── train.py        # Training script
├── tests/              # Unit tests
├── requirements.txt
└── README.md
```
