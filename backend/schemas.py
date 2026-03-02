from pydantic import BaseModel


class ChatIn(BaseModel):
    message: str
    session_id: str = None


class AuthRegisterIn(BaseModel):
    username: str
    password: str


class AuthLoginIn(BaseModel):
    username: str
    password: str
