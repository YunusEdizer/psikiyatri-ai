"""Kimlik doğrulama ve kullanıcı yönetimi uç noktaları.

- ``POST /auth/login``  : kullanıcı adı + parola → JWT erişim belirteci
- ``GET  /auth/me``     : oturum sahibinin bilgisi
- ``POST /auth/users``  : yeni kullanıcı (yalnızca yönetici)
- ``GET  /auth/users``  : kullanıcı listesi (yalnızca yönetici)
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data import audit
from ..db.database import get_db
from ..db.models import Role, User
from .schemas import TokenResponse, UserCreate, UserOut
from .security import (
    create_access_token,
    get_current_user,
    hash_password,
    require_roles,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["kimlik"])


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
    )


# Basit bellek-içi giriş hız-sınırı (kaba-kuvvet parola denemelerine karşı):
# kullanıcı başına, kayan pencerede en fazla _MAX_FAILS başarısız deneme.
_LOGIN_FAILS: dict[str, list[float]] = {}
_MAX_FAILS = 5
_WINDOW_SEC = 300.0


def _too_many_fails(username: str) -> bool:
    now = time.monotonic()
    fails = [t for t in _LOGIN_FAILS.get(username, []) if now - t < _WINDOW_SEC]
    _LOGIN_FAILS[username] = fails
    return len(fails) >= _MAX_FAILS


@router.post("/login", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
) -> TokenResponse:
    if _too_many_fails(form.username):
        audit.log_event("login_rate_limited", detail={"username": form.username})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Çok fazla başarısız deneme. Birkaç dakika sonra tekrar deneyin.",
            headers={"Retry-After": str(int(_WINDOW_SEC))},
        )
    user = db.scalar(select(User).where(User.username == form.username))
    if user is None or not user.is_active or not verify_password(form.password, user.password_hash):
        _LOGIN_FAILS.setdefault(form.username, []).append(time.monotonic())
        audit.log_event("login_failed", detail={"username": form.username})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya parola hatalı.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _LOGIN_FAILS.pop(form.username, None)  # başarı → sayacı sıfırla
    audit.log_event("login", detail={"username": user.username, "role": user.role})
    return TokenResponse(
        access_token=create_access_token(user.username, user.role),
        username=user.username,
        full_name=user.full_name,
        role=user.role,
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(Role.YONETICI)),
) -> UserOut:
    if db.scalar(select(User).where(User.username == payload.username)) is not None:
        raise HTTPException(status_code=409, detail="Bu kullanıcı adı zaten kayıtlı.")
    user = User(
        username=payload.username,
        full_name=payload.full_name,
        role=payload.role,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    audit.log_event(
        "user_created",
        detail={"username": user.username, "role": user.role, "by": admin.username},
    )
    return _user_out(user)


@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_roles(Role.YONETICI)),
) -> list[UserOut]:
    users = db.scalars(select(User).order_by(User.id)).all()
    return [_user_out(u) for u in users]


__all__ = ["router"]
