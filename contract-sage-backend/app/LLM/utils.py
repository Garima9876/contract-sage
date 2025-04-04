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


# utils.py
import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def clean_and_split(document: str) -> List[str]:
    """Clean a legal document and split it into sentences.
    
    Args:
        document: The raw legal document text
        
    Returns:
        List of cleaned sentences
    """
    ensure_nltk_resources()
    
    # Remove case citations
    document = re.sub(r'\(.*?\d{4}.*?\)', '', document)
    
    # Normalize whitespace
    document = re.sub(r'\s+', ' ', document).strip()
    
    # Split into sentences
    try:
        sentences = sent_tokenize(document)
    except Exception as e:
        logger.warning(f"Error during sentence tokenization: {e}. Falling back to simple splitting.")
        sentences = document.split('. ')
    
    # Remove empty sentences
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def assemble_prompt(segmented_doc: List[Dict[str, str]], 
                    named_entities: Dict[str, List[str]], 
                    focus: Optional[List[str]] = None) -> str:
    """Assemble a prompt for the LLM based on segmented document and entities.
    
    Args:
        segmented_doc: List of dictionaries containing sentences and their segments
        named_entities: Dictionary mapping entity types to lists of entity values
        focus: Optional list of segment types to focus on
        
    Returns:
        Formatted prompt string
    """
    from config import config
    
    # Group sentences by segment type
    sections = {label: [] for label in config.SEGMENT_LABELS}
    for item in segmented_doc:
        segments = sections.get(item['segment'], [])
        segments.append(item['sentence'])
        sections[item['segment']] = segments
    
    # Build the prompt
    prompt = 'You are a legal AI assistant. Summarize the legal case below.\n\n'
    
    # Add content from each section
    for label in config.SEGMENT_LABELS:
        if focus and label not in focus:
            continue
        if sections[label]:
            prompt += f'### {label}:\n' + ' '.join(sections[label]) + '\n\n'
    
    # Add named entities
    prompt += '### Named Entities Extracted:\n'
    for entity, values in named_entities.items():
        unique_values = list(set(values))  # Remove duplicates
        prompt += f'- **{entity}**: {", ".join(unique_values)}\n'
    
    prompt += '\nProvide a comprehensive summary that captures the key aspects of the case.\nOutput:'
    
    return prompt


def extract_key_points(summary_text: str) -> Dict[str, Any]:
    """Extract key points from the generated summary.
    
    Args:
        summary_text: The raw summary text
        
    Returns:
        Dictionary with full summary and extracted bullet points
    """
    # Extract bullet points if present
    bullets = re.findall(r'(?:•|\*|-)\s*(.+)', summary_text)
    
    return {
        'summary': summary_text,
        'key_points': bullets if bullets else []
    }




