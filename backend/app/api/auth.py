"""Authentication and user registration API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import create_access_token, get_current_user, hash_password, verify_password
from app.db.repositories import count_users, create_user, get_user_by_username, log_audit
from app.models.schemas import TokenResponse, UserLogin, UserRegister

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(payload: UserRegister) -> dict:
    if get_user_by_username(payload.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    # First user can self-register as admin; subsequent registrations default to investigator unless admin-created
    role = payload.role if count_users() == 0 else (payload.role if payload.role != "admin" else "investigator")
    user = create_user(
        username=payload.username,
        full_name=payload.full_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=role,
    )
    log_audit("USER_REGISTER", "user", str(user["id"]), f"Registered role={role}", user["id"], user["username"])
    return {"user": user}


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin) -> TokenResponse:
    user = get_user_by_username(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    if not user.get("is_active", 1):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    token = create_access_token(subject=user["username"], role=user["role"], user_id=user["id"])
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    log_audit("USER_LOGIN", "user", str(user["id"]), "Login successful", user["id"], user["username"])
    return TokenResponse(access_token=token, user=safe_user)


@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)) -> dict:
    return {"user": current_user}
