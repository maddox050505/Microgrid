from fastapi import FastAPI

app = FastAPI(title="Microgrid API")

@app.get("/health")
def health():
    return {"ok": True}

from uuid import uuid4
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from .database import Base, engine, get_db
from .models import Company, User, Plan
from .security import hash_password, verify_password, create_token
from .deps import get_current_user

Base.metadata.create_all(bind=engine)

