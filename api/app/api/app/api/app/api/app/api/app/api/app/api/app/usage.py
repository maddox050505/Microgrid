from uuid import uuid4
from sqlalchemy.orm import Session
from .models import UsageEvent

def log_usage(db: Session, company_id: str, event_type: str, quantity: int = 1) -> None:
    evt = UsageEvent(
        id=str(uuid4()),
        company_id=company_id,
        event_type=event_type,
        quantity=quantity,
    )
    db.add(evt)
    db.commit()
