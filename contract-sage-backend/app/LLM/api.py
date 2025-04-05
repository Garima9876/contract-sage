import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
import asyncio
from contextlib import asynccontextmanager
from .config import config
import sys
import os
sys.path.append(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
processing_tasks = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from .model import model_manager
    from .phrase_anomaly import anomaly_detector
    
    logger.info(f'Initializing with device: {config.device}')
    if config.device == 'mps' and hasattr(torch.backends, 'mps'):
        torch.mps.empty_cache()
    
    logger.info('Preloading tokenizers and models...')
    try:
        # Load existing models
        _ = model_manager.seg_tokenizer
        _ = model_manager.seg_model
        _ = model_manager.llm_tokenizer
        _ = model_manager.llm_model
        _ = model_manager.ner_tokenizer
        _ = model_manager.ner_model
        
        # Initialize anomaly detector
        _ = anomaly_detector._get_model()
        _ = anomaly_detector._initialize_db()
        
        logger.info('All models loaded successfully')
    except Exception as e:
        logger.error(f'Failed to preload models: {e}')
        raise
    
    yield
    
    logger.info('Shutting down API')
    if hasattr(torch, 'mps') and config.device == 'mps':
        torch.mps.empty_cache()
    elif config.device == 'cuda':
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class SummarizeRequest(BaseModel):
    text: str
    focus_sections: Optional[List[str]] = None

class NERRequest(BaseModel):
    text: str

class AnomalyDetectionRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.8

class SegmentRequest(BaseModel):
    text: str

class TaskStatus(BaseModel):
    status: str
    task_id: str
    result: Optional[Dict[str, Any]] = None

# Original endpoints
@app.post('/summarize')
async def summarize_file(file: UploadFile=File(...)):
    try:
        content = (await file.read()).decode('utf-8')
        if len(content) < 50:
            raise HTTPException(status_code=400, detail='Document too short')
        from .summarizer import summarize_legal_text
        result = summarize_legal_text(content)
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        return result
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail='File must be valid UTF-8 text')
    except Exception as e:
        logger.error(f'Error processing file: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/summarize/text')
async def summarize_text(request: SummarizeRequest):
    try:
        from .summarizer import summarize_legal_text
        
        if len(request.text) < 50:
            raise HTTPException(status_code=400, detail='Text too short')
        
        result = summarize_legal_text(request.text, focus_sections=request.focus_sections)
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
    except Exception as e:
        logger.error(f'Error processing text: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add new API for NER only
@app.post('/ner')
async def extract_ner(request: NERRequest):
    try:
        from .ner import extract_named_entities
        
        if len(request.text) < 10:
            raise HTTPException(status_code=400, detail='Text too short')
        
        logger.info('Extracting named entities')
        entities = extract_named_entities(request.text)
        
        return {
            'entities': entities, 
            'metadata': {
                'entity_counts': {k: len(v) for k, v in entities.items()}, 
                'total_entities': sum((len(v) for v in entities.values()))
            }
        }
    except Exception as e:
        logger.error(f'Error extracting named entities: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add new API for segmentation only
@app.post('/segment')
async def segment_document(request: SegmentRequest):
    try:
        from .segmenter import segment_legal_document
        
        if len(request.text) < 50:
            raise HTTPException(status_code=400, detail='Text too short')
        
        logger.info('Segmenting document')
        segmented_doc, named_entities = segment_legal_document(request.text)
        
        if not segmented_doc:
            raise HTTPException(status_code=500, detail='Document segmentation failed')
        
        return {
            'segments': segmented_doc,
            'entities': named_entities,
            'metadata': {
                'segment_counts': {
                    label: sum((1 for item in segmented_doc if item['segment'] == label)) 
                    for label in set((item['segment'] for item in segmented_doc))
                },
                'total_segments': len(segmented_doc),
                'entity_counts': {k: len(v) for k, v in named_entities.items()},
            }
        }
    except Exception as e:
        logger.error(f'Error segmenting document: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add new API for anomaly detection
@app.post('/anomaly/detect')
async def detect_anomalies(request: AnomalyDetectionRequest):
    try:
        from .phrase_anomaly import anomaly_detector
        from .utils import clean_and_split
        
        if len(request.text) < 50:
            raise HTTPException(status_code=400, detail='Text too short')
        
        # Split the text into sentences
        sentences = clean_and_split(request.text)
        
        if not sentences:
            raise HTTPException(status_code=400, detail='Could not extract sentences from text')
        
        # Detect anomalies
        result = anomaly_detector.detect_anomalies(sentences, threshold=request.threshold)
        
        return {
            'anomalies': result['anomalies'],
            'stats': {
                'total_phrases': len(sentences),
                'anomalous_phrases': len(result['anomalies']),
                'avg_score': result['avg_score'],
                'max_score': result['max_score'],
                'min_score': result['min_score'],
            }
        }
    except Exception as e:
        logger.error(f'Error detecting anomalies: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add API for adding to phrase database
@app.post('/anomaly/add')
async def add_phrases(request: AnomalyDetectionRequest):
    try:
        from .phrase_anomaly import anomaly_detector
        from .utils import clean_and_split
        
        if len(request.text) < 50:
            raise HTTPException(status_code=400, detail='Text too short')
        
        # Split the text into sentences
        sentences = clean_and_split(request.text)
        
        if not sentences:
            raise HTTPException(status_code=400, detail='Could not extract sentences from text')
        
        # Add phrases to database
        anomaly_detector.add_to_db(sentences)
        
        return {
            'status': 'success',
            'phrases_added': len(sentences),
            'total_phrases_in_db': len(anomaly_detector.phrases)
        }
    except Exception as e:
        logger.error(f'Error adding phrases to database: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Process pipeline steps asynchronously
async def process_segmentation_async(task_id: str, content: str):
    try:
        from .segmenter import segment_legal_document
        
        processing_tasks[task_id]['status'] = 'processing'
        segmented_doc, named_entities = segment_legal_document(content)
        
        if not segmented_doc:
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = 'Document segmentation failed'
            return
        
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = {
            'segments': segmented_doc,
            'entities': named_entities,
            'metadata': {
                'segment_counts': {
                    label: sum((1 for item in segmented_doc if item['segment'] == label)) 
                    for label in set((item['segment'] for item in segmented_doc))
                },
                'total_segments': len(segmented_doc),
                'entity_counts': {k: len(v) for k, v in named_entities.items()},
            }
        }
    except Exception as e:
        logger.error(f'Error in segmentation task: {e}', exc_info=True)
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

async def process_ner_async(task_id: str, content: str):
    try:
        from .ner import extract_named_entities
        
        processing_tasks[task_id]['status'] = 'processing'
        entities = extract_named_entities(content)
        
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = {
            'entities': entities,
            'metadata': {
                'entity_counts': {k: len(v) for k, v in entities.items()},
                'total_entities': sum((len(v) for v in entities.values()))
            }
        }
    except Exception as e:
        logger.error(f'Error in NER task: {e}', exc_info=True)
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

async def process_anomaly_detection_async(task_id: str, content: str, threshold: float = 0.8):
    try:
        from .phrase_anomaly import anomaly_detector
        from .utils import clean_and_split
        
        processing_tasks[task_id]['status'] = 'processing'
        
        # Split into sentences
        sentences = clean_and_split(content)
        
        if not sentences:
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = 'Could not extract sentences from text'
            return
        
        # Detect anomalies
        result = anomaly_detector.detect_anomalies(sentences, threshold=threshold)
        
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = {
            'anomalies': result['anomalies'],
            'stats': {
                'total_phrases': len(sentences),
                'anomalous_phrases': len(result['anomalies']),
                'avg_score': result['avg_score'],
                'max_score': result['max_score'],
                'min_score': result['min_score'],
            }
        }
    except Exception as e:
        logger.error(f'Error in anomaly detection task: {e}', exc_info=True)
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

async def process_document_async(task_id: str, content: str, focus_sections: Optional[List[str]]=None):
    try:
        from .summarizer import summarize_legal_text
        
        processing_tasks[task_id]['status'] = 'processing'
        result = summarize_legal_text(content, focus_sections=focus_sections)
        
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = result
    except Exception as e:
        logger.error(f'Error in background task: {e}', exc_info=True)
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

# Add async endpoints for each pipeline step
@app.post('/segment/async')
async def segment_async(request: SegmentRequest, background_tasks: BackgroundTasks):
    task_id = f'segment_{int(time.time())}_{id(request)}'
    processing_tasks[task_id] = {'status': 'queued', 'created_at': time.time(), 'text_length': len(request.text)}
    background_tasks.add_task(process_segmentation_async, task_id=task_id, content=request.text)
    return {'task_id': task_id, 'status': 'queued', 'message': 'Segmentation started'}

@app.post('/ner/async')
async def ner_async(request: NERRequest, background_tasks: BackgroundTasks):
    task_id = f'ner_{int(time.time())}_{id(request)}'
    processing_tasks[task_id] = {'status': 'queued', 'created_at': time.time(), 'text_length': len(request.text)}
    background_tasks.add_task(process_ner_async, task_id=task_id, content=request.text)
    return {'task_id': task_id, 'status': 'queued', 'message': 'NER extraction started'}

@app.post('/anomaly/async')
async def anomaly_async(request: AnomalyDetectionRequest, background_tasks: BackgroundTasks):
    task_id = f'anomaly_{int(time.time())}_{id(request)}'
    processing_tasks[task_id] = {'status': 'queued', 'created_at': time.time(), 'text_length': len(request.text)}
    background_tasks.add_task(process_anomaly_detection_async, task_id=task_id, content=request.text, threshold=request.threshold)
    return {'task_id': task_id, 'status': 'queued', 'message': 'Anomaly detection started'}

@app.post('/summarize/async')
async def summarize_async(request: SummarizeRequest, background_tasks: BackgroundTasks):
    task_id = f'summarize_{int(time.time())}_{id(request)}'
    processing_tasks[task_id] = {'status': 'queued', 'created_at': time.time(), 'text_length': len(request.text)}
    background_tasks.add_task(process_document_async, task_id=task_id, content=request.text, focus_sections=request.focus_sections)
    return {'task_id': task_id, 'status': 'queued', 'message': 'Summarization started'}

@app.get('/task/{task_id}')
async def get_task_status(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail='Task not found')
    
    task_info = processing_tasks[task_id]
    response = {'task_id': task_id, 'status': task_info['status'], 'created_at': task_info['created_at']}
    
    if task_info['status'] == 'completed':
        response['result'] = task_info['result']
    elif task_info['status'] == 'failed':
        response['error'] = task_info.get('error', 'Unknown error')
    
    return response

@app.get('/health')
async def health_check():
    from .model import model_manager
    from .phrase_anomaly import anomaly_detector
    
    try:
        assert model_manager.llm_model is not None
        assert model_manager.seg_model is not None
        assert model_manager.ner_model is not None
        assert anomaly_detector.model is not None
        
        return {
            'status': 'healthy', 
            'device': config.device,
            'models': {
                'llm': True,
                'segmentation': True,
                'ner': True,
                'anomaly_detection': True
            }
        }
    except Exception as e:
        logger.error(f'Health check failed: {e}')
        raise HTTPException(status_code=500, detail='Service unavailable')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)