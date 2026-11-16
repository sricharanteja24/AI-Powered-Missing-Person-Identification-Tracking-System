"""Pydantic v2 request and response schemas."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=8, max_length=128)
    role: Literal["admin", "investigator", "reviewer"] = "investigator"


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class CaseCreate(BaseModel):
    case_id: str = Field(..., min_length=2, max_length=64)
    full_name: str = Field(..., min_length=2, max_length=120)
    age: int | None = Field(default=None, ge=0, le=120)
    gender: str | None = Field(default=None, max_length=32)
    last_known_location: str = Field(..., min_length=2, max_length=200)
    last_seen_date: str = Field(..., min_length=4, max_length=32)
    description: str | None = Field(default=None, max_length=1000)


class AlertVerificationRequest(BaseModel):
    verification_status: Literal["verified_candidate", "rejected_candidate", "pending_review"]
    verification_notes: str | None = Field(default=None, max_length=1000)
