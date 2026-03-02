import base64
import hmac
import hashlib
import os
import re
import secrets
from fastapi import HTTPException, status

PASSWORD_PBKDF2_ITERATIONS = int(
    os.environ.get("AUTH_PASSWORD_PBKDF2_ITERATIONS", "200000")
)


def validate_username(username: str) -> str:
    cleaned = (username or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,32}", cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名需为3-32位，仅支持字母/数字/._-",
        )
    return cleaned


def validate_password(password: str) -> str:
    cleaned = password or ""
    if len(cleaned) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密码长度至少6位",
        )
    return cleaned


def hash_password(password: str, salt_b64: str | None = None):
    if salt_b64:
        salt = base64.b64decode(salt_b64.encode("ascii"))
    else:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_PBKDF2_ITERATIONS,
    )
    return (
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, salt_b64: str, expected_hash_b64: str) -> bool:
    _, actual_hash = hash_password(password, salt_b64=salt_b64)
    return hmac.compare_digest(actual_hash, expected_hash_b64)
