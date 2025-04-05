from typing import Dict, Any, List, Optional
import logging
logger = logging.getLogger(__name__)

def summarize_legal_text(document: str, focus_sections: Optional[List[str]]=None, detect_anomalies: bool=True) -> Dict[str, Any]:
    try:
        logger.info('Starting legal document summarization')
        from .segmenter import segment_legal_document
        from .utils import assemble_prompt, extract_key_points
        from .generator import generate_summary
        from .phrase_anomaly import anomaly_detector
        
        if not document or len(document) < 50:
            return {'error': 'Document too short or empty'}
        
        logger.info('Segmenting document')
        segmented_doc, named_entities = segment_legal_document(document)
        
        if not segmented_doc:
            return {'error': 'Document segmentation failed'}
        
        # Process anomaly detection if requested
        anomalies = {}
        if detect_anomalies:
            try:
                logger.info('Detecting phrase anomalies')
                # Get sentences from segmented document
                sentences = [item['sentence'] for item in segmented_doc]
                
                # Detect anomalies
                anomaly_results = anomaly_detector.detect_anomalies(sentences)
                
                # Add results to metadata
                anomalies = {
                    'detected': anomaly_results['anomalies'],
                    'count': len(anomaly_results['anomalies']),
                    'avg_score': anomaly_results['avg_score']
                }
                
                # If not already in the database, add these phrases
                if anomaly_detector.index.ntotal < 100:  # Only add if DB is small
                    logger.info('Adding phrases to anomaly detection database')
                    anomaly_detector.add_to_db(sentences)
                
            except Exception as e:
                logger.warning(f'Anomaly detection failed: {e}')
                anomalies = {'error': str(e)}
        
        logger.info('Assembling prompt')
        prompt = assemble_prompt(segmented_doc, named_entities, focus=focus_sections)
        
        logger.info('Generating summary')
        raw_summary = generate_summary(prompt)
        
        if raw_summary.startswith('Error generating summary'):
            return {'error': raw_summary}
        
        result = extract_key_points(raw_summary)
        result['metadata'] = {
            'segment_counts': {
                label: sum((1 for item in segmented_doc if item['segment'] == label)) 
                for label in set((item['segment'] for item in segmented_doc))
            }, 
            'entity_counts': {
                entity: len(values) for entity, values in named_entities.items()
            }, 
            'total_sentences': len(segmented_doc), 
            'focused_sections': focus_sections
        }
        
        # Add anomaly results to metadata if available
        if anomalies:
            result['metadata']['anomalies'] = anomalies
        
        logger.info('Summarization complete')
        return result
    except Exception as e:
        logger.error(f'Error in summarization pipeline: {e}', exc_info=True)
        return {'error': str(e)}