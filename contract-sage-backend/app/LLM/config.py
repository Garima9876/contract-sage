# # === config.py ===
# import torch
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# from peft import PeftModel
# from model import LegalSemanticSegmentation

# class ModelConfig:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

#         # Load segmentation model
#         self.seg_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
#         seg_base_model = AutoModel.from_pretrained("law-ai/InLegalBERT")
#         self.classifier_model = LegalSemanticSegmentation(seg_base_model).to(self.device)

#         # Load summarization model
#         self.llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#         base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
#         self.llm_model = PeftModel.from_pretrained(base_model, "ajay-drew/Mistral-7B-Indian-Law").to(self.device)

# # Singleton config instance
# model_config = ModelConfig()


# # === config.py ===
# import torch
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification
# from peft import PeftModel
# from model import LegalSemanticSegmentation

# class ModelConfig:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

#         # Load segmentation model (InLegalBERT)
#         self.seg_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
#         seg_base_model = AutoModel.from_pretrained("law-ai/InLegalBERT")
#         self.classifier_model = LegalSemanticSegmentation(seg_base_model).to(self.device)

#         # Load summarization model (Mistral with LoRA Fine-Tuning)
#         self.llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#         base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
#         self.llm_model = PeftModel.from_pretrained(base_model, "ajay-drew/Mistral-7B-Indian-Law").to(self.device)

#         # Load Named Entity Recognition (NER) model (Fine-tuned XLM-RoBERTa)
#         self.ner_tokenizer = AutoTokenizer.from_pretrained("your-ner-model-path")
#         self.ner_model = AutoModelForTokenClassification.from_pretrained("your-ner-model-path").to(self.device)

# # Singleton config instance
# model_config = ModelConfig()

# config.py
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AppConfig:
    """Application configuration with model paths and parameters."""
    # Model paths
    SEGMENTATION_MODEL_PATH: str = 'law-ai/InLegalBERT'
    LLM_MODEL_PATH: str = 'mistralai/Mistral-7B-v0.1'
    LLM_ADAPTER_PATH: str = 'ajay-drew/Mistral-7B-Indian-Law'
    NER_MODEL_PATH: str = os.environ.get('NER_MODEL_PATH', 'your-ner-model-path')
    
    # Device configuration
    USE_CUDA: bool = True
    USE_MPS: bool = True
    
    # Processing parameters
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    BATCH_SIZE: int = 8
    
    # Segmentation labels
    SEGMENT_LABELS: List[str] = ['FACTS', 'ARGUMENTS', 'STATUTE', 'PRECEDENT', 'RATIO', 'RULING', 'OTHER']
    
    @property
    def device(self) -> str:
        """Determine the appropriate device for tensor computations."""
        import torch
        if self.USE_CUDA and torch.cuda.is_available():
            return 'cuda'
        elif self.USE_MPS and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

# Create a global config instance
config = AppConfig()


