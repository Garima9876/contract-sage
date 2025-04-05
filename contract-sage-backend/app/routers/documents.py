# app/routers/documents.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.schemas.document import DocumentRead
from app.models.document import Document
from app.database import SessionLocal, engine, Base
from typing import List
from sqlalchemy.orm import Session
import shutil
import os
from fastapi import Form

router = APIRouter()

# Ensure tables exist (for demo purposes)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

UPLOAD_DIR = "./uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/upload", response_model=DocumentRead)
async def upload_document(
    file: UploadFile = File(...),
    focus_sections: str = Form(""),  # Comma-separated list; optional
    db: Session = Depends(get_db)
):
    # Save the file to disk
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract text from file; for now, we assume the file is plain text (UTF-8)
    try:
        with open(file_location, "rb") as f:
            content = f.read().decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    
    if len(content.strip()) < 50:
        raise HTTPException(status_code=400, detail="Document too short")
    
    # Process focus sections if provided (split comma-separated values)
    focus = focus_sections.split(",") if focus_sections else None
    
    # ----- Call LLM functions sequentially -----
    try:
        # Segmentation
        from app.LLM.segmenter import segment_legal_document
        segmented_doc, segmentation_entities = segment_legal_document(content)
        if not segmented_doc:
            raise HTTPException(status_code=500, detail="Document segmentation failed")
        
        # Named Entity Recognition
        from app.LLM.ner import extract_named_entities
        ner_entities = extract_named_entities(content)
        
        # Anomaly Detection
        from app.LLM.utils import clean_and_split
        from app.LLM.phrase_anomaly import anomaly_detector
        sentences = clean_and_split(content)
        if not sentences:
            raise HTTPException(status_code=400, detail="Could not extract sentences from text")
        anomaly_result = anomaly_detector.detect_anomalies(sentences, threshold=0.5)
        
        # Summarization
        # from app.LLM.summarizer import summarize_legal_text
        # summary_result = summarize_legal_text(content, focus_sections=focus)
        # if "error" in summary_result:
        #     raise HTTPException(status_code=500, detail=summary_result["error"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in LLM processing: {str(e)}")
    
    # Combine all LLM results into a single dictionary for frontend display
    combined_result = {
        "segmentation": {
            "segments": segmented_doc,
            "entities": segmentation_entities,
            "segment_counts": {
                label: sum(1 for item in segmented_doc if item["segment"] == label)
                for label in set(item["segment"] for item in segmented_doc)
            },
            "total_segments": len(segmented_doc),
        },
        "named_entities": ner_entities,
        "anomalies": {
            "anomalies": anomaly_result.get("anomalies", []),
            "stats": anomaly_result.get("stats", {})
        },
        # "summary": summary_result.get("summary", "No summary available"),
    }
    
    # Create a new document record with the summary (you might choose to store the combined result as JSON in another field)
    # new_doc = Document(
    #     filename=file.filename,
    #     summary=combined_result.get("summary", "No summary available"),
    #     user_id=1  # Replace with actual authenticated user ID when implementing authentication
    # )
    # db.add(new_doc)
    # db.commit()
    # db.refresh(new_doc)
    
    # Create a document record
    new_doc = Document(
        filename=file.filename,
        summary="Summary not generated",  # Just placeholder
        user_id=1  # Dummy user ID
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    return JSONResponse(status_code=200, content={
        "document": {
            "id": new_doc.id,
            "filename": new_doc.filename,
            "summary": new_doc.summary
        },
        "llm_results": combined_result
    })
    
    # Optionally, you could update new_doc with the combined result before returning it,
    # or simply return the combined_result along with new_doc.id.
    # Here, we return the document record (per your DocumentRead schema) and the LLM outputs.
    
    # For example, if DocumentRead has extra fields (or you modify it to include them), you can do:
    # new_doc.combined_result = combined_result
    # db.commit()
    
    # For now, we'll return a JSON with both the document record and combined LLM results.
    return JSONResponse(status_code=200, content={
        "document": {
            "id": new_doc.id,
            "filename": new_doc.filename,
            "summary": new_doc.summary
        },
        "llm_results": combined_result
    })
    
@router.get("/documents", response_model=List[DocumentRead])
async def get_all_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).all()
    return documents

@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
