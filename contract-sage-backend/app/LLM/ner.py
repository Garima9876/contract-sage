# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import torch
# from config import NER_MODEL_PATH, DEVICE

# # Load tokenizer and model once to avoid redundant initialization
# ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
# ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH).to(DEVICE)

# def extract_named_entities(text):
#     """
#     Extract named entities from the given text using the fine-tuned XLM-RoBERTa model.
#     """
#     ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=0 if DEVICE == "cuda" else -1)
#     ner_results = ner_pipeline(text)
    
#     extracted_entities = {}
#     for entity in ner_results:
#         label = entity["entity"]
#         value = entity["word"]
#         extracted_entities.setdefault(label, []).append(value)
    
#     return extracted_entities

# ner.py
from transformers import pipeline
import torch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from legal text.
    
    Args:
        text: The legal document text
        
    Returns:
        Dictionary mapping entity types to lists of entity values
    """
    from .config import config
    from .model import model_manager
    
    try:
        # Initialize NER pipeline
        ner_pipeline = pipeline(
            'ner', 
            model=model_manager.ner_model, 
            tokenizer=model_manager.ner_tokenizer, 
            device=0 if config.device == 'cuda' else -1
        )
        
        # Process text in chunks if it's very long
        max_length = model_manager.ner_tokenizer.model_max_length
        if len(text) > max_length * 2:
            logger.info(f"Text length {len(text)} exceeds model capacity, processing in chunks")
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            all_results = []
            for chunk in chunks:
                all_results.extend(ner_pipeline(chunk))
            ner_results = all_results
        else:
            ner_results = ner_pipeline(text)
        
        # Group entities by type
        extracted_entities = {}
        for entity in ner_results:
            label = entity['entity']
            value = entity['word']
            extracted_entities.setdefault(label.replace('B-', '').replace('I-', ''), []).append(value)
            
        return extracted_entities
        
    except Exception as e:
        logger.error(f"Error extracting named entities: {e}")
        return {}





