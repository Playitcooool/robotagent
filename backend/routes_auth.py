import json
import secrets
import time
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException, status

from backend.schemas import AuthLoginIn, AuthRegisterIn


def register_auth_routes(
    app,
    *,
    get_auth_redis: Callable[[], object],
    auth_session_ttl_seconds: int,
    auth_user_key: Callable[[str], str],
    auth_session_key: Callable[[str], str],
    validate_username: Callable[[str], str],
    validate_password: Callable[[str], str],
    hash_password: Callable[[str], tuple[str, str]],
    verify_password: Callable[[str, str, str], bool],
):
    async def require_auth_user(
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        auth_redis = get_auth_redis()
        if auth_redis is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="认证服务未就绪",
            )

        header = (authorization or "").strip()
        if not header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="缺少认证令牌",
            )

        token = header[7:].strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌无效",
            )

        session_raw = await auth_redis.get(auth_session_key(token))
        if not session_raw:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登录已过期，请重新登录",
            )

        try:
            session = json.loads(session_raw)
            if not isinstance(session, dict):
                raise ValueError("invalid session")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登录会话无效",
            )

        await auth_redis.expire(auth_session_key(token), auth_session_ttl_seconds)
        return {
            "token": token,
            "uid": session.get("uid"),
            "username": session.get("username"),
        }

    @app.post("/api/auth/register")
    async def auth_register(payload: AuthRegisterIn):
        auth_redis = get_auth_redis()
        if auth_redis is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="认证服务未就绪",
            )

        username = validate_username(payload.username)
        password = validate_password(payload.password)
        user_key = auth_user_key(username)
        existing = await auth_redis.get(user_key)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="用户名已存在",
            )

        salt_b64, password_hash_b64 = hash_password(password)
        created_at = time.time()
        user_doc = {
            "uid": secrets.token_hex(12),
            "username": username,
            "password_salt": salt_b64,
            "password_hash": password_hash_b64,
            "created_at": created_at,
        }
        await auth_redis.set(user_key, json.dumps(user_doc, ensure_ascii=False))

        token = secrets.token_urlsafe(32)
        session_doc = {
            "uid": user_doc["uid"],
            "username": user_doc["username"],
            "created_at": created_at,
        }
        await auth_redis.setex(
            auth_session_key(token),
            auth_session_ttl_seconds,
            json.dumps(session_doc, ensure_ascii=False),
        )
        return {
            "token": token,
            "user": {"uid": user_doc["uid"], "username": user_doc["username"]},
            "expires_in": auth_session_ttl_seconds,
        }

    @app.post("/api/auth/login")
    async def auth_login(payload: AuthLoginIn):
        auth_redis = get_auth_redis()
        if auth_redis is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="认证服务未就绪",
            )

        username = validate_username(payload.username)
        password = validate_password(payload.password)
        user_raw = await auth_redis.get(auth_user_key(username))
        if not user_raw:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
            )
        try:
            user_doc = json.loads(user_raw)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="用户数据损坏",
            )

        ok = verify_password(
            password,
            user_doc.get("password_salt", ""),
            user_doc.get("password_hash", ""),
        )
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
            )

        token = secrets.token_urlsafe(32)
        session_doc = {
            "uid": user_doc["uid"],
            "username": user_doc["username"],
            "created_at": time.time(),
        }
        await auth_redis.setex(
            auth_session_key(token),
            auth_session_ttl_seconds,
            json.dumps(session_doc, ensure_ascii=False),
        )
        return {
            "token": token,
            "user": {"uid": user_doc["uid"], "username": user_doc["username"]},
            "expires_in": auth_session_ttl_seconds,
        }

    @app.get("/api/auth/me")
    async def auth_me(current_user: dict = Depends(require_auth_user)):
        return {
            "user": {
                "uid": current_user.get("uid"),
                "username": current_user.get("username"),
            }
        }

    @app.post("/api/auth/logout")
    async def auth_logout(current_user: dict = Depends(require_auth_user)):
        auth_redis = get_auth_redis()
        if auth_redis is not None:
            await auth_redis.delete(auth_session_key(current_user.get("token", "")))
        return {"ok": True}

    return require_auth_user
