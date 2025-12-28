from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Company, User, Plan, Role
from .schemas import CompanyCreate, LoginIn
from .security import hash_password, verify_password, create_access_token
from .deps import get_current_user, get_company
from .plans import rules_for
from .usage import log_usage

app = FastAPI(title="Microgrid API")

Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"ok": True}

# -----------------------
# Auth / tenancy (Layer A)
# -----------------------
@app.post("/companies")
def create_company(payload: CompanyCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.admin_email.lower().strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use")

    company = Company(
        id=str(uuid4()),
        name=payload.company_name.strip(),
        plan=Plan(payload.plan),
        seats_allowed=1 if payload.plan == "base" else 3,
        sites_allowed=1 if payload.plan == "base" else (10 if payload.plan == "pro" else 50),
    )
    db.add(company)

    admin = User(
        id=str(uuid4()),
        company_id=company.id,
        email=payload.admin_email.lower().strip(),
        password_hash=hash_password(payload.admin_password),
        role=Role.admin,
    )
    db.add(admin)
    db.commit()

    token = create_access_token({"sub": admin.id, "company_id": company.id, "role": admin.role.value})
    return {"access_token": token}

@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower().strip(), User.is_active == True).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.id, "company_id": user.company_id, "role": user.role.value})
    return {"access_token": token}

@app.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return {"email": user.email, "company_id": user.company_id, "role": user.role.value}

# -----------------------
# Example monetized endpoint wrapper pattern
# (You will wrap your real optimize endpoint like this)
# -----------------------
@app.post("/run")
def run_paid_compute(
    payload: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    company = get_company(db, user.company_id)
    rules = rules_for(company.plan.value)

    # Metering (Layer C)
    log_usage(db, company.id, "analysis", 1)

    # TODO: call your existing optimization logic here
    return {"status": "ok", "plan": company.plan.value, "rules": rules.__dict__}
