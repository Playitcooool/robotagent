"""
Retry utility with exponential backoff and jitter.

Inspired by Claude Code's services/api/withRetry.ts
"""

import asyncio
import functools
import random
import time
from typing import Callable, Optional, Tuple, TypeVar

T = TypeVar('T')

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5  # seconds
DEFAULT_MAX_DELAY = 32.0  # seconds
DEFAULT_JITTER_FACTOR = 0.25  # 25% randomization


def calculate_retry_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
    retry_after: Optional[float] = None,
) -> float:
    """
    Calculate delay for retry attempt with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter_factor: Randomization factor (0.25 = 25%)
        retry_after: Optional Retry-After header value in seconds

    Returns:
        Delay in seconds before next retry
    """
    # Honor Retry-After header if present
    if retry_after is not None:
        return retry_after

    # Exponential backoff: baseDelay * 2^(attempt)
    delay = min(base_delay * (2 ** attempt), max_delay)

    # Add jitter: up to jitter_factor randomization
    jitter = random.random() * jitter_factor * delay
    return delay + jitter


async def with_retry_async(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retryable_exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs,
) -> T:
    """
    Async retry wrapper with exponential backoff and jitter.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called before each retry (attempt, exception)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e

            if attempt >= max_retries:
                print(f"[Retry] All {max_retries + 1} attempts failed")
                raise

            delay = calculate_retry_delay(attempt, base_delay, max_delay)

            if on_retry:
                on_retry(attempt + 1, e)

            print(f"[Retry] attempt {attempt + 1}/{max_retries + 1} failed: {e}")
            print(f"[Retry] waiting {delay:.2f}s before retry...")

            await asyncio.sleep(delay)

    # All retries exhausted (should not reach here if retryable_exceptions covers all failures)
    if last_exception:
        raise last_exception


def with_retry_sync(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retryable_exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs,
) -> T:
    """
    Sync retry wrapper with exponential backoff and jitter.

    Args:
        func: Synchronous function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called before each retry (attempt, exception)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e

            if attempt >= max_retries:
                print(f"[Retry] All {max_retries + 1} attempts failed")
                raise

            delay = calculate_retry_delay(attempt, base_delay, max_delay)

            if on_retry:
                on_retry(attempt + 1, e)

            print(f"[Retry] attempt {attempt + 1}/{max_retries + 1} failed: {e}")
            print(f"[Retry] waiting {delay:.2f}s before retry...")

            time.sleep(delay)

    # All retries exhausted (should not reach here if retryable_exceptions covers all failures)
    if last_exception:
        raise last_exception


def retry_async(max_retries: int = DEFAULT_MAX_RETRIES, **retry_kwargs):
    """
    Decorator for async functions with retry support.

    Usage:
        @retry_async(max_retries=3)
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Merge kwargs with retry_kwargs, but max_retries is always explicit
            merged = {**retry_kwargs, **kwargs}
            return await with_retry_async(
                func, *args,
                max_retries=max_retries,
                **merged
            )
        return wrapper
    return decorator


def retry_sync(max_retries: int = DEFAULT_MAX_RETRIES, **retry_kwargs):
    """
    Decorator for sync functions with retry support.

    Usage:
        @retry_sync(max_retries=3)
        def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Merge kwargs with retry_kwargs, but max_retries is always explicit
            merged = {**retry_kwargs, **kwargs}
            return with_retry_sync(
                func, *args,
                max_retries=max_retries,
                **merged
            )
        return wrapper
    return decorator
