# segmenter.py
import torch
from typing import List, Dict, Tuple, Any
import logging
from config import config
from model import model_manager
from utils import clean_and_split, get_lemmatized_text, get_pos_features
from ner import extract_named_entities
    

logger = logging.getLogger(__name__)

def segment_legal_document(document: str) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:

    sentences = clean_and_split(document)
    if not sentences:
        logger.warning('No sentences found after preprocessing')
        return ([], {})
    
    # Enhance with lemmatization
    lemmatized_sentences = [get_lemmatized_text(s) for s in sentences]
    
    # Get POS features for additional context
    pos_features = [get_pos_features(s) for s in sentences]
    
    logger.info('Extracting named entities')
    named_entities = extract_named_entities(document)
    
    # Extract named entities
    logger.info("Extracting named entities")
    named_entities = extract_named_entities(document)
    
    # Set model to evaluation mode
    model_manager.seg_model.eval()
    
    results = []
    batch_size = config.BATCH_SIZE
    
    logger.info(f"Processing {len(sentences)} sentences in batches of {batch_size}")
    
    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        
        # Tokenize batch
        inputs = model_manager.seg_tokenizer(
            batch_sentences, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=512
        ).to(config.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model_manager.seg_model(**inputs)
        
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Map predictions to labels
        for j, sentence in enumerate(batch_sentences):
            segment_idx = predictions[j]
            segment_label = config.SEGMENT_LABELS[segment_idx] if segment_idx < len(config.SEGMENT_LABELS) else 'OTHER'
            results.append({'sentence': sentence, 'segment': segment_label})
    
    logger.info(f"Document segmentation complete, found {len(results)} segments")
    return results, named_entities

