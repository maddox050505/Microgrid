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

@app.post("/companies")
def create_company(
    name: str,
    admin_email: str,
    admin_password: str,
    plan: Plan = Plan.base,
    db: Session = Depends(get_db),
):
    company = Company(
        id=str(uuid4()),
        name=name,
        plan=plan,
    )
    db.add(company)

    admin = User(
        id=str(uuid4()),
        company_id=company.id,
        email=admin_email.lower(),
        password_hash=hash_password(admin_password),
        role="admin",
    )
    db.add(admin)
    db.commit()

    token = create_token({"sub": admin.id, "company_id": company.id})
    return {"access_token": token}

@app.post("/auth/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token({"sub": user.id, "company_id": user.company_id})
    return {"access_token": token}

@app.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return {
        "email": user.email,
        "company_id": user.company_id,
        "role": user.role,
    }

