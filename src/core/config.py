"""
Configuration for Bias Detector.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BiasLabel(str, Enum):
    """Bias categories detected by the model."""
    GENDER = "gender"
    AGE = "age"
    CULTURE = "culture"


@dataclass
class BiasDetectorConfig:
    """Configuration for the bias detection pipeline."""
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    classifier_type: str = "logistic"
    
    # Labels
    bias_labels: list[BiasLabel] = field(default_factory=lambda: [
        BiasLabel.GENDER,
        BiasLabel.AGE,
        BiasLabel.CULTURE,
    ])
    
    # MLflow
    experiment_name: str = "bias-detector-ml"
    
    def get_label_names(self) -> list[str]:
        return [label.value for label in self.bias_labels]
