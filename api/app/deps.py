from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from .database import get_db
from .models import User
from .security import decode_token

bearer = HTTPBearer()

def get_current_user(
    creds = Depends(bearer),
    db: Session = Depends(get_db),
):
    try:
        payload = decode_token(creds.credentials)
        user_id = payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(
        User.id == user_id,
        User.is_active == True,
    ).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
