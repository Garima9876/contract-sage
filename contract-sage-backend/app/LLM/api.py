# # === app.py (example FastAPI endpoint) ===
# from fastapi import FastAPI, UploadFile, File
# from summarizer import summarize_legal_text

# app = FastAPI()

# @app.post("/summarize")
# async def summarize_file(file: UploadFile = File(...)):
#     content = (await file.read()).decode("utf-8")
#     result = summarize_legal_text(content, focus_sections=["FACTS", "RATIO", "RULING"])
#     return result
# @app.post("/ner")
# def get_named_entities(request: DocumentRequest):
#     """
#     API endpoint to extract named entities from a given legal document.
#     """
#     if not request.text.strip():
#         raise HTTPException(status_code=400, detail="Empty document provided.")
    
#     entities = extract_named_entities(request.text)
#     return {"named_entities": entities}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
import asyncio
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store background tasks
processing_tasks = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load resources
    from model import model_manager
    logger.info("Preloading tokenizers...")
    # Just access the properties to trigger loading
    _ = model_manager.seg_tokenizer
    _ = model_manager.llm_tokenizer
    _ = model_manager.ner_tokenizer
    logger.info("API startup complete")
    yield
    # Shutdown: Clean up resources
    logger.info("Shutting down API")


app = FastAPI(lifespan=lifespan)

class SummarizeRequest(BaseModel):
    text: str
    focus_sections: Optional[List[str]] = None


class TaskStatus(BaseModel):
    status: str
    task_id: str
    result: Optional[Dict[str, Any]] = None


@app.post('/summarize')
async def summarize_file(file: UploadFile = File(...)):
    """Endpoint to summarize a legal document from a file upload."""
    try:
        content = (await file.read()).decode('utf-8')
        
        if len(content) < 50:
            raise HTTPException(status_code=400, detail="Document too short")
            
        from summarizer import summarize_legal_text
        result = summarize_legal_text(content)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/summarize/text')
async def summarize_text(request: SummarizeRequest):
    """Endpoint to summarize legal document text directly."""
    try:
        from summarizer import summarize_legal_text
        result = summarize_legal_text(request.text, focus_sections=request.focus_sections)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_async(task_id: str, content: str, focus_sections: Optional[List[str]] = None):
    """Background task for processing large documents."""
    try:
        from summarizer import summarize_legal_text
        processing_tasks[task_id]["status"] = "processing"
        result = summarize_legal_text(content, focus_sections=focus_sections)
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result"] = result
    except Exception as e:
        logger.error(f"Error in background task: {e}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


@app.post('/summarize/async')
async def summarize_async(request: SummarizeRequest, background_tasks: BackgroundTasks):
    """Asynchronous endpoint for processing large documents."""
    task_id = f"task_{int(time.time())}_{id(request)}"
    
    processing_tasks[task_id] = {
        "status": "queued",
        "created_at": time.time()
    }
    
    background_tasks.add_task(
        process_document_async, 
        task_id=task_id,
        content=request.text,
        focus_sections=request.focus_sections
    )
    
    return {"task_id": task_id, "status": "queued"}


@app.get('/task/{task_id}')
async def get_task_status(task_id: str):
    """Check status of an asynchronous task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_info = processing_tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task_info["status"]
    }
    
    if task_info["status"] == "completed":
        response["result"] = task_info["result"]
    elif task_info["status"] == "failed":
        response["error"] = task_info.get("error", "Unknown error")
        
    return response


@app.get('/health')
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}