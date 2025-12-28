from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from .database import get_db
from .models import User, Company
from .security import decode_token

bearer = HTTPBearer(auto_error=False)

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if not creds:
        raise HTTPException(status_code=401, detail="Missing auth token")

    try:
        payload = decode_token(creds.credentials)
        user_id = payload.get("sub")
        company_id = payload.get("company_id")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(
        User.id == user_id,
        User.company_id == company_id,
        User.is_active == True,
    ).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user

def get_company(db: Session, company_id: str) -> Company:
    company = db.query(Company).filter(Company.id == company_id, Company.is_active == True).first()
    if not company:
        raise HTTPException(status_code=403, detail="Company missing or inactive")
    return company
