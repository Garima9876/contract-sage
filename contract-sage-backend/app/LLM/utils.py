# # === utils.py ===
# import re
# import nltk
# from nltk.tokenize import sent_tokenize
# import torch

# SEGMENT_LABELS = ["FACTS", "ARGUMENTS", "STATUTE", "PRECEDENT", "RATIO", "RULING", "OTHER"]

# nltk.download("punkt")

# def clean_and_split(document):
#     document = re.sub(r"\(.*?\d{4}.*?\)", "", document)
#     document = re.sub(r"\s+", " ", document).strip()
#     sentences = sent_tokenize(document)
#     return [s.strip() for s in sentences if len(s.strip()) > 0]

# # def segment_legal_document(document, model, tokenizer, device):
# #     sentences = clean_and_split(document)
# #     model.eval().to(device)
# #     results = []
# #     batch_size = 8
# #     for i in range(0, len(sentences), batch_size):
# #         batch = sentences[i:i+batch_size]
# #         inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
# #         with torch.no_grad():
# #             outputs = model(**inputs)
# #         predictions = torch.argmax(outputs, dim=1).cpu().numpy()
# #         for j, sentence in enumerate(batch):
# #             results.append({"sentence": sentence, "segment": SEGMENT_LABELS[predictions[j]]})
# #     return results

# from config import seg_model, seg_tokenizer, DEVICE
# from ner import extract_named_entities
# from utils import clean_and_split
# import torch

# SEGMENT_LABELS = ["FACTS", "ARGUMENTS", "STATUTE", "PRECEDENT", "RATIO", "RULING", "OTHER"]

# def segment_legal_document(document: str):
#     sentences = clean_and_split(document)
#     seg_model.eval().to(DEVICE)
#     results = []
#     batch_size = 8
    
#     # Extract named entities using NER module
#     named_entities = extract_named_entities(document)

#     for i in range(0, len(sentences), batch_size):
#         batch_sentences = sentences[i:i+batch_size]
#         inputs = seg_tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)
        
#         with torch.no_grad():
#             outputs = seg_model(**inputs)
        
#         predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
#         for j, sentence in enumerate(batch_sentences):
#             segment_label = SEGMENT_LABELS[predictions[j]]
#             results.append({"sentence": sentence, "segment": segment_label})
    
#     return results, named_entities

# def assemble_prompt(segmented_doc, focus=None):
#     sections = {label: [] for label in SEGMENT_LABELS}
#     for item in segmented_doc:
#         sections[item["segment"]].append(item["sentence"])
#     prompt = "You are a legal AI assistant. Summarize the legal case below.\n\n"
#     for label in SEGMENT_LABELS:
#         if focus and label not in focus:
#             continue
#         if sections[label]:
#             prompt += f"### {label}:\n" + " ".join(sections[label]) + "\n\n"
#     prompt += "Output:"
#     return prompt

# def generate_text_with_mistral(prompt, model, tokenizer, max_new_tokens=512, temperature=0.7):
#     tokenizer.padding_side = "left"
#     tokenizer.pad_token = tokenizer.eos_token
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=temperature,
#             top_p=0.95,
#             top_k=50,
#             repetition_penalty=1.1
#         )
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# def extract_key_points(summary_text):
#     bullets = re.findall(r"\u2022\s*(.+)", summary_text)
#     return {"summary": summary_text, "bullets": bullets or [summary_text]}


# 1. First, let's modify api.py to remove the import and call:
import re
import os
from typing import List, Dict, Any, Optional
import logging
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from transformers import pipeline  # We'll use a transformer-based POS tagger
logger = logging.getLogger(__name__)

# Initialize a transformer-based POS tagger pipeline
pos_tagger = None

def ensure_nltk_resources():
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        logger.warning(f'Failed to download NLTK resources: {e}')

def initialize_pos_tagger():
    """Initialize the transformer-based POS tagger"""
    global pos_tagger
    if pos_tagger is None:
        try:
            # Using a small, efficient model for POS tagging
            pos_tagger = pipeline("token-classification", 
                                model="vblagoje/bert-english-uncased-finetuned-pos", 
                                aggregation_strategy="simple")
            logger.info("Initialized transformer-based POS tagger")
        except Exception as e:
            logger.error(f"Failed to initialize POS tagger: {e}")
            raise

def clean_and_split(document: str) -> List[str]:
    ensure_nltk_resources()
    document = re.sub(r'\(.*?\d{4}.*?\)', '', document)
    document = re.sub(r'\s+', ' ', document).strip()
    try:
        sentences = sent_tokenize(document)
    except Exception as e:
        logger.warning(f'Error during sentence tokenization: {e}. Falling back to simple splitting.')
        sentences = document.split('. ')
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def assemble_prompt(segmented_doc: List[Dict[str, str]], named_entities: Dict[str, List[str]], focus: Optional[List[str]]=None) -> str:
    from config import config
    sections = {label: [] for label in config.SEGMENT_LABELS}
    for item in segmented_doc:
        segments = sections.get(item['segment'], [])
        segments.append(item['sentence'])
        sections[item['segment']] = segments
    prompt = 'You are a legal AI assistant. Summarize the legal case below.\n\n'
    for label in config.SEGMENT_LABELS:
        if focus and label not in focus:
            continue
        if sections[label]:
            prompt += f'### {label}:\n' + ' '.join(sections[label]) + '\n\n'
    prompt += '### Named Entities Extracted:\n'
    for entity, values in named_entities.items():
        unique_values = list(set(values))
        prompt += f'- **{entity}**: {", ".join(unique_values)}\n'
    prompt += '\nProvide a comprehensive summary that captures the key aspects of the case.\nOutput:'
    return prompt

def extract_key_points(summary_text: str) -> Dict[str, Any]:
    bullets = re.findall(r'(?:•|\*|-)\s*(.+)', summary_text)
    return {'summary': summary_text, 'key_points': bullets if bullets else []}

def get_lemmatized_text(text):
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def get_pos_features(text):
    """Get POS features using transformer-based POS tagger"""
    try:
        if pos_tagger is None:
            initialize_pos_tagger()
        
        # Process the text with the transformer-based POS tagger
        pos_results = pos_tagger(text)
        
        # Count POS tags
        pos_counts = {}
        for item in pos_results:
            tag = item['entity_group']
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        
        return pos_counts
    except Exception as e:
        logger.error(f"Error in POS tagging: {e}")
        return {}