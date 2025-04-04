# app/main.py
from fastapi import FastAPI
# from app.routers import auth, documents

app = FastAPI()

# app.include_router(auth.router, prefix="/api/auth")
# app.include_router(documents.router, prefix="/api/documents")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
