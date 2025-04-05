# app/routers/auth.py
from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.user import UserCreate, UserRead
from app.models.user import User
from app.database import SessionLocal, engine, Base
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
import uuid
import jwt
import time
import asyncio

conf = ConnectionConfig(
    MAIL_USERNAME="diptimahakalkar21@gmail.com",
    MAIL_PASSWORD="ferr zhmu zegy ktdm",  # Use App Password, not your Gmail password
    MAIL_FROM="diptimahakalkar21@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

# Ensure tables exist (for demo purposes)
Base.metadata.create_all(bind=engine)

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your_secret_key_here"  # Replace with your secret key
ALGORITHM = "HS256"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def send_activation_email(email: EmailStr, activation_link: str):
    message = MessageSchema(
        subject="Activate Your Contract Sage Account",
        recipients=[email],
        body=(
            f"Dear User,\n\n"
            f"Thank you for registering with Contract Sage.\n\n"
            f"To activate your account, please click the link below:\n"
            f"{activation_link}\n\n"
            f"If you did not register for this account, please ignore this email.\n\n"
            f"Best regards,\n"
            f"The Contract Sage Team"
        ),
        subtype="plain"
    )

    fm = FastMail(conf)
    await fm.send_message(message)

class RegisterRequest(BaseModel):
    email: EmailStr

@router.post("/register")
async def register(payload: RegisterRequest):
    email = payload.email

    token_data = {
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    activation_jwt = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    activation_link = f"http://localhost:5173/auth/activate?token={activation_jwt}"
    await send_activation_email(email, activation_link)
    
    return {"message": "Activation link sent to your email"}

@router.post("/activate")
def activate_user(token: str, password: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("email")
        if email is None:
            raise HTTPException(status_code=400, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Activation link expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=400, detail="Invalid activation token")

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Account already activated")

    hashed_password = pwd_context.hash(password)
    new_user = User(
        email=email,
        hashed_password=hashed_password,
        is_active=True,
        activation_token=None
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Account activated successfully"}

@router.post("/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not db_user.is_active:
        raise HTTPException(status_code=403, detail="Account not activated")

    token_data = {
        "sub": db_user.email,
        "id": db_user.id,
        "exp": time.time() + 3600
    }
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}
