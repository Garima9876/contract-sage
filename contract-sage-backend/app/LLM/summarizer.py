from typing import Dict, Any, List, Optional
import logging
logger = logging.getLogger(__name__)

def summarize_legal_text(document: str, focus_sections: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        logger.info('Starting legal document summarization')
        
        from .segmenter import segment_legal_document
        from .utils import assemble_prompt, extract_key_points
        from .generator import generate_summary
        
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
