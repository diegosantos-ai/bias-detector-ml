# Detector de Viés em RH (ML)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)](https://mlflow.org/)

Modelo de Aprendizado de Máquina para detectar vieses implícitos (gênero, idade, cultura) em descrições de vagas. Construído com Sentence-Transformers e rastreado com MLflow.

## 🚀 Funcionalidades Principais

- **Classificação Multi-rótulo**: Detecta múltiplos tipos de viés simultaneamente.
- **Explicabilidade**: Categoriza o tipo de viés (ex: "Etarismo", "Viés de Gênero").
- **Integração com MLflow**: Rastreamento completo de experimentos e registro de modelos.
- **API REST**: Endpoint FastAPI para análise em tempo real.

## 🛠️ Tecnologias

- **ML**: `scikit-learn`, `sentence-transformers`
- **Ops**: `mlflow`, `docker`
- **API**: `fastapi`

## 📦 Instalação

```bash
# Clonar
git clone https://github.com/seususuario/bias-detector-ml.git
cd bias-detector-ml

# Ambiente Virtual
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Dependências
pip install -r requirements.txt
```

## 🏃‍♂️ Início Rápido

1. **Iniciar MLflow (Opcional)**
   ```bash
   mlflow ui --port 5001
   ```

2. **Treinar Modelo**
   ```bash
   # Gera dados sintéticos e treina o modelo
   python -m src.train
   ```
   *Nota: A primeira execução fará o download do modelo de embeddings (80MB).*

3. **Executar API**
   ```bash
   uvicorn src.api.app:app --reload
   ```
   Acesse a documentação em: http://localhost:8000/docs

## 🧪 Testes

```bash
# Executar testes unitários
pytest tests/
```

## 📂 Estrutura do Projeto

```
bias-detector-ml/
├── src/
│   ├── api/            # Endpoints da API
│   ├── data/           # Geração de dados
│   ├── ml/             # Pipeline de ML
│   └── train.py        # Script de treinamento
├── tests/              # Testes unitários
├── requirements.txt
└── README.md
```
