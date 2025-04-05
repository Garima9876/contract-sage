# # # === summarizer.py ===
# # # from config import model_config
# # # from utils import segment_legal_document, assemble_prompt, generate_text_with_mistral, extract_key_points

# # # def summarize_legal_text(document: str, focus_sections=None):
# # #     segmented = segment_legal_document(
# # #         document,
# # #         model_config.classifier_model,
# # #         model_config.seg_tokenizer,
# # #         model_config.device
# # #     )
# # #     prompt = assemble_prompt(segmented, focus=focus_sections)
# # #     summary = generate_text_with_mistral(
# # #         prompt,
# # #         model_config.llm_model,
# # #         model_config.llm_tokenizer
# # #     )
# # #     return extract_key_points(summary)

# # from config import model_config
# # from ner import extract_named_entities
# # from utils import segment_legal_document, assemble_prompt, generate_text_with_mistral, extract_key_points

# # def summarize_legal_doc(document: str, classifier_model, classifier_tokenizer, llm_model, llm_tokenizer, focus_sections=None):
# #     """
# #     1. Extract Named Entities from document
# #     2. Perform segmentation
# #     3. Integrate Named Entities into the prompt
# #     4. Summarize using Mistral
# #     """
    
# #     # Step 1: Extract Named Entities
# #     named_entities = extract_named_entities(document)

# #     # Step 2: Perform Legal Document Segmentation
# #     segmented = segment_legal_document(document, classifier_model, classifier_tokenizer)

# #     # Step 3: Create a summary prompt with extracted entities
# #     prompt = assemble_prompt(segmented, focus=focus_sections)

# #     # Append Named Entities to provide context for the LLM
# #     prompt += "\n\n### Named Entities Extracted:\n"
# #     for entity, values in named_entities.items():
# #         prompt += f"- **{entity}**: {', '.join(values)}\n"

# #     # Step 4: Generate Summary
# #     summary = generate_text_with_mistral(prompt, llm_model, llm_tokenizer)
    
# #     return summary

# # summarizer.py
# from typing import Dict, Any, List, Optional
# import logging

# logger = logging.getLogger(__name__)

# def summarize_legal_text(document: str, focus_sections: Optional[List[str]] = None) -> Dict[str, Any]:
#     """Main function to process and summarize a legal document.
    
#     Args:
#         document: The raw legal document text
#         focus_sections: Optional list of sections to focus on
        
#     Returns:
#         Dictionary with summary results
#     """
#     try:
#         logger.info("Starting legal document summarization")
        
#         from segmenter import segment_legal_document
#         from utils import assemble_prompt, extract_key_points
#         from generator import generate_summary
        
#         # Check document length
#         if not document or len(document) < 50:
#             return {"error": "Document too short or empty"}
        
#         # Segment document and extract entities
#         logger.info("Segmenting document")
#         segmented_doc, named_entities = segment_legal_document(document)
        
#         # Assemble prompt for LLM
#         logger.info("Assembling prompt")
#         prompt = assemble_prompt(segmented_doc, named_entities, focus=focus_sections)
        
#         # Generate summary
#         raw_summary = generate_summary(prompt)
        
#         # Extract key points
#         result = extract_key_points(raw_summary)
        
#         # Add metadata
#         result["metadata"] = {
#             "segment_counts": {label: sum(1 for item in segmented_doc if item["segment"] == label) 
#                               for label in set(item["segment"] for item in segmented_doc)},
#             "entity_counts": {entity: len(values) for entity, values in named_entities.items()},
#             "total_sentences": len(segmented_doc),
#             "focused_sections": focus_sections
#         }
        
#         logger.info("Summarization complete")
#         return result
        
#     except Exception as e:
#         logger.error(f"Error in summarization pipeline: {e}")
#         return {"error": str(e)}

from typing import Dict, Any, List, Optional
import logging
logger = logging.getLogger(__name__)

def summarize_legal_text(document: str, focus_sections: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        logger.info('Starting legal document summarization')
        
        from segmenter import segment_legal_document
        from utils import assemble_prompt, extract_key_points
        from generator import generate_summary
        
        if not document or len(document) < 50:
            return {'error': 'Document too short or empty'}
        
        logger.info('Segmenting document')
        segmented_doc, named_entities = segment_legal_document(document)
        
        if not segmented_doc:
            return {'error': 'Document segmentation failed'}
        
        logger.info('Assembling prompt')
        prompt = assemble_prompt(segmented_doc, named_entities, focus=focus_sections)
        
        logger.info('Generating summary')
        raw_summary = generate_summary(prompt)
        
        if raw_summary.startswith('Error generating summary'):
            return {'error': raw_summary}
        
        result = extract_key_points(raw_summary)
        result['metadata'] = {
            'segment_counts': {
                label: sum(1 for item in segmented_doc if item['segment'] == label)
                for label in set(item['segment'] for item in segmented_doc)
            },
            'entity_counts': {
                entity: len(values) for entity, values in named_entities.items()
            },
            'total_sentences': len(segmented_doc),
            'focused_sections': focus_sections
        }
        
        logger.info('Summarization complete')
        return result
        
    except Exception as e:
        logger.error(f'Error in summarization pipeline: {e}', exc_info=True)
        return {'error': str(e)}
