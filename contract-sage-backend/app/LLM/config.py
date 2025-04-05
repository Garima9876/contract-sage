from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class AppConfig:
    SEGMENTATION_MODEL_PATH: str = 'law-ai/InLegalBERT'
    LLM_MODEL_PATH: str = 'varma007ut/Indian_Legal_Assitant'
    LLM_ADAPTER_PATH: str = None
    NER_MODEL_PATH: str = 'FacebookAI/xlm-roberta-base'
    POS_TAGGER_MODEL: str = 'vblagoje/bert-english-uncased-finetuned-pos'
    USE_CUDA: bool = True
    USE_MPS: bool = True
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    BATCH_SIZE: int = 8
    SEGMENT_LABELS: List[str] = field(default_factory=lambda: ['FACTS', 'ARGUMENTS', 'STATUTE', 'PRECEDENT', 'RATIO', 'RULING', 'OTHER'])
    OFFLOAD_FOLDER = os.getenv('OFFLOAD_FOLDER', './offload')
    
    # Anomaly detection configurations
    EMBEDDING_MODEL_PATH: str = 'all-MiniLM-L6-v2'
    ANOMALY_THRESHOLD: float = 0.8
    VECTOR_DB_PATH: str = os.getenv('VECTOR_DB_PATH', './vector_db')
    
    @property
    def device(self) -> str:
        import torch
        if self.USE_CUDA and torch.cuda.is_available():
            return 'cuda'
        elif self.USE_MPS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    @property
    def device_map(self) -> str:
        return 'auto' if self.USE_CUDA or self.USE_MPS else 'cpu'
        
config = AppConfig()