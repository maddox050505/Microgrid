from pydantic import BaseModel, EmailStr
from typing import Literal

Role = Literal["admin", "sales", "viewer"]
Plan = Literal["base", "pro", "enterprise"]

class CompanyCreate(BaseModel):
    company_name: str
    admin_email: EmailStr
    admin_password: str
    plan: Plan = "base"

class LoginIn(BaseModel):
    email: EmailStr
    password: str
