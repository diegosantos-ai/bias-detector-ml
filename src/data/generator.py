"""
Módulo de geração de dados usando listas de palavras e modelos (templates).
"""
import random
import pandas as pd
from typing import List, Dict

# Listas de palavras
GENDER_BIASED = [
    "ninja", "rockstar", "guerreiro", "campeão", "dominador",
    "agressivo", "ambicioso", "competitivo", "líder nato"
]

AGE_BIASED = [
    "jovem dinâmico", "recém-formado", "sangue novo", "digital native",
    "energia de sobra", "ambiente universitário", "primeiro emprego",
    "fome de aprender", "nativo digital"
]

CULTURE_BIASED = [
    "happy hour obrigatório", "work hard play hard", "fit cultural",
    "vestir a camisa", "viciado em trabalho", "disponibilidade total",
    "ambiente informal", "cerveja liberada"
]

NEUTRAL_TERMS = [
    "profissional", "colaborador", "especialista", "analista",
    "experiência", "conhecimento", "habilidade", "formação",
    "responsável", "organizado", "comunicativo"
]

TEMPLATES = [
    "Procuramos um {term} para nosso time.",
    "Buscamos alguém {term} para a vaga.",
    "Você é {term}? Venha trabalhar conosco!",
    "Requisitos: ser {term}.",
    "Ambiente {term} e desafiador.",
    "Se você é {term}, essa vaga é sua.",
    "Precisamos de um perfil {term}.",
]

def generate_synthetic_data(samples: int = 500) -> pd.DataFrame:
    """
    Gera conjunto de dados sintético para detecção de viés.
    
    Retorna:
        pd.DataFrame com colunas ['text', 'gender', 'age', 'culture']
    """
    data = []
    
    for _ in range(samples):
        # Decidir tipos de viés para esta amostra (aleatoriamente)
        has_gender = random.random() < 0.3
        has_age = random.random() < 0.3
        has_culture = random.random() < 0.3
        
        # Selecionar termo baseado no viés
        terms = []
        if has_gender: terms.append(random.choice(GENDER_BIASED))
        if has_age: terms.append(random.choice(AGE_BIASED))
        if has_culture: terms.append(random.choice(CULTURE_BIASED))
        
        # Se não houver viés, selecionar termo neutro
        if not terms:
            term = random.choice(NEUTRAL_TERMS)
        else:
            term = " e ".join(terms)
        
        # Gerar texto
        template = random.choice(TEMPLATES)
        text = template.format(term=term)
        
        # Adicionar
        data.append({
            "text": text,
            "gender": 1 if has_gender else 0,
            "age": 1 if has_age else 0,
            "culture": 1 if has_culture else 0
        })
        
    return pd.DataFrame(data)
