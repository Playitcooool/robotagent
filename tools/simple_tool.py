from __future__ import annotations

from functools import wraps


def tool(*decorator_args, response_format: str = "content", **_decorator_kwargs):
    """Compatibility decorator for legacy callable tool functions.

    Production tool registration now lives in the TypeScript Pi gateway. This
    keeps old experiments and scripts callable without importing LangChain just
    to attach basic tool metadata.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.name = func.__name__
        wrapper.description = func.__doc__ or ""
        wrapper.response_format = response_format
        wrapper.args_schema = None
        return wrapper

    if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
        return decorate(decorator_args[0])
    return decorate
