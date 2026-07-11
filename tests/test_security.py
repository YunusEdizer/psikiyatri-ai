"""Parola özetleme ve JWT testleri."""

import pytest
from fastapi import HTTPException

from psikiyatri_ai.api.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_hash_roundtrip():
    h = hash_password("gizli-parola-1")
    assert h.startswith("pbkdf2_sha256$")
    assert verify_password("gizli-parola-1", h)


def test_wrong_password_rejected():
    h = hash_password("dogru-parola")
    assert not verify_password("yanlis-parola", h)


def test_same_password_different_salt():
    assert hash_password("ayni") != hash_password("ayni")


def test_malformed_hash_rejected():
    assert not verify_password("x", "bozuk-format")
    assert not verify_password("x", "")


def test_token_roundtrip():
    token = create_access_token("dr.ayse", "psikiyatrist")
    payload = decode_token(token)
    assert payload["sub"] == "dr.ayse"
    assert payload["role"] == "psikiyatrist"


def test_expired_token_rejected():
    token = create_access_token("dr.ayse", "psikolog", expires_minutes=-1)
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_tampered_token_rejected():
    token = create_access_token("dr.ayse", "psikolog")
    with pytest.raises(HTTPException):
        decode_token(token[:-4] + "aaaa")
