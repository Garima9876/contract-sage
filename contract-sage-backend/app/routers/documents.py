from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.schemas.document import DocumentRead
from app.models.document import Document
from app.database import SessionLocal, engine, Base
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
    focus_sections: str = Form(""),  # Declare as a parameter here
    db: Session = Depends(get_db)
):
    # Save file to UPLOAD_DIR
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract text from file. For simplicity, we assume the file is plain text.
    try:
        with open(file_location, "rb") as f:
            content = f.read().decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    
    # Validate file length before processing
    if len(content) < 50:
        raise HTTPException(status_code=400, detail="Document too short")
    
    # Now focus_sections is a string from the form; we can split it
    focus = focus_sections.split(",") if focus_sections else None
    
    # Use the summarizer function from your LLM module to process the content
    from app.LLM.summarizer import summarize_legal_text
    result = summarize_legal_text(content, focus_sections=focus)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Create a new document record with the generated summary.
    new_doc = Document(
        filename=file.filename,
        summary=result.get("summary", "No summary available"),
        user_id=1  # Replace with the authenticated user's ID as needed
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    # Return the document record including the summary.
    return new_doc


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
