# app/schemas/document.py
from pydantic import BaseModel
from typing import Optional

class DocumentUpload(BaseModel):
    filename: str

class DocumentRead(BaseModel):
    id: int
    filename: str
    summary: Optional[str] = None

    class Config:
        from_attributes = True
