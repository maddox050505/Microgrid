from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Company, User, Plan
from .security import hash_password, verify_password, create_token
from .deps import get_current_user, get_company
from .plans import rules_for

app = FastAPI(title="Microgrid API")

# Create tables at startup (simple MVP). Later we can move to Alembic migrations.
Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"ok": True}


# -----------------------
# Layer 1: Company + Auth
# -----------------------
@app.post("/companies")
def create_company(
    name: str,
    admin_email: str,
    admin_password: str,
    plan: Plan = Plan.base,
    db: Session = Depends(get_db),
):
    # prevent duplicate user emails
    existing = db.query(User).filter(User.email == admin_email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use")

    company = Company(
        id=str(uuid4()),
        name=name.strip(),
        plan=plan,
    )
    db.add(company)

    admin = User(
        id=str(uuid4()),
        company_id=company.id,
        email=admin_email.lower().strip(),
        password_hash=hash_password(admin_password),
        role="admin",
    )
    db.add(admin)
    db.commit()

    token = create_token({"sub": admin.id, "company_id": company.id})
    return {"access_token": token, "company_id": company.id, "plan": company.plan.value}


@app.post("/auth/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email.lower().strip(), User.is_active == True).first()
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


# -----------------------
# Layer 1: Plan enforcement on a paid endpoint
# -----------------------
@app.post("/run")
def run_paid_compute(
    payload: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Attach request to the company (multi-tenant boundary)
    company = get_company(db, user.company_id)

    # Pull plan rules (feature gating begins here)
    plan_str = company.plan.value if hasattr(company.plan, "value") else str(company.plan)
    rules = rules_for(plan_str)

    # For now: just enforce "must have runs > 0" as a placeholder gate.
    # Next step is Layer 2: metering + monthly limits.
    if rules.runs_per_month <= 0:
        raise HTTPException(status_code=402, detail="Upgrade required")

    # TODO: Replace this with your real optimization call.
    # Keep this endpoint name stable; the frontend calls /run.
    result = {"status": "ok", "echo": payload}

    return {
        "result": result,
        "plan": plan_str,
        "limits": {"runs_per_month": rules.runs_per_month},
        "features": {
            "exports_enabled": rules.exports_enabled,
            "portfolio_enabled": rules.portfolio_enabled,
        },
    }

