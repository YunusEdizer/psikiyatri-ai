"""Kimlik doğrulama ve yetkilendirme.

- Parola özeti: stdlib PBKDF2-SHA256 (harici bağımlılık yok).
- Erişim belirteci: JWT (HS256, `PAI_SECRET_KEY` ile imzalı).
- Roller: yönetici / psikiyatrist / psikolog (`db.models.Role`).

KVKK/GDPR bağlamı: hasta verisine erişen tüm uç noktalar bu modüldeki
bağımlılıklarla korunur; erişimler denetim günlüğüne kullanıcı adıyla
işlenir (hesap verebilirlik).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.database import get_db
from ..db.models import Role, User
from ..logging_conf import logger

_ALGORITHM = "HS256"
_PBKDF2_ITERATIONS = 200_000

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# --- Parola özetleme -------------------------------------------------------


def hash_password(password: str) -> str:
    """PBKDF2-SHA256 ile tuzlu parola özeti üretir."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Parolayı kayıtlı özetle sabit-zamanlı karşılaştırır."""
    try:
        _scheme, iterations, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), expected)
    except (ValueError, TypeError):
        return False


# --- JWT -------------------------------------------------------------------


def create_access_token(username: str, role: str, expires_minutes: int | None = None) -> str:
    settings = get_settings()
    minutes = expires_minutes or settings.access_token_expire_minutes
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(UTC) + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict:
    """Belirteci doğrular; geçersiz/süresi dolmuşsa 401 üretir."""
    try:
        return jwt.decode(token, get_settings().secret_key, algorithms=[_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz veya süresi dolmuş oturum. Yeniden giriş yapın.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# --- FastAPI bağımlılıkları --------------------------------------------------


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    payload = decode_token(token)
    user = db.scalar(select(User).where(User.username == payload.get("sub", "")))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı bulunamadı veya devre dışı.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_roles(*roles: Role):
    """Belirtilen rollerden birini zorunlu kılan bağımlılık üretir."""
    allowed = {r.value for r in roles}

    def _checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Bu işlem için yetkiniz yok (gereken rol: {', '.join(sorted(allowed))}).",
            )
        return user

    return _checker


# --- İlk kullanıcı tohumu ----------------------------------------------------


def ensure_admin_user(db: Session) -> None:
    """Kullanıcı tablosu boşsa yapılandırmadaki yönetici hesabını oluşturur."""
    if db.scalar(select(User).limit(1)) is not None:
        return
    settings = get_settings()
    db.add(
        User(
            username=settings.admin_username,
            full_name="Sistem Yöneticisi",
            role=Role.YONETICI.value,
            password_hash=hash_password(settings.admin_password),
        )
    )
    db.commit()
    logger.warning(
        "İlk yönetici hesabı oluşturuldu: '{}'. Üretimde parolayı hemen değiştirin.",
        settings.admin_username,
    )


__all__ = [
    "create_access_token",
    "decode_token",
    "ensure_admin_user",
    "get_current_user",
    "hash_password",
    "require_roles",
    "verify_password",
]
