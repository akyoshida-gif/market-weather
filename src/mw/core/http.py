"""HTTP helpers: per-host token-bucket rate limiting and retry with backoff (DESIGN §5)."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlparse

import httpx

T = TypeVar("T")

DEFAULT_UA = "market-weather/0.1 (personal research; non-redistributed)"


class RateLimiter:
    """Thread-safe token bucket: at most `rate` acquisitions per second, burst `burst`."""

    def __init__(self, rate: float, burst: int = 1) -> None:
        self.rate = rate
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def limiter_for(host: str, rate: float, burst: int = 1) -> RateLimiter:
    with _limiters_lock:
        if host not in _limiters:
            _limiters[host] = RateLimiter(rate, burst)
        return _limiters[host]


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call fn with exponential backoff + jitter. Re-raises the last error."""
    for i in range(attempts):
        try:
            return fn()
        except retry_on:
            if i == attempts - 1:
                raise
            delay = min(max_delay, base_delay * 2**i) * (0.5 + random.random())
            time.sleep(delay)
    raise AssertionError("unreachable")


class HttpClient:
    """httpx wrapper applying per-host rate limits and retries on transient errors."""

    def __init__(
        self,
        rate_per_host: float = 1.0,
        user_agent: str = DEFAULT_UA,
        timeout: float = 60.0,
    ) -> None:
        self.rate = rate_per_host
        self.client = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=timeout, follow_redirects=True
        )

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        host = urlparse(url).netloc
        limiter = limiter_for(host, self.rate)

        def _do() -> httpx.Response:
            limiter.acquire()
            resp = self.client.get(url, **kwargs)  # type: ignore[arg-type]
            if resp.status_code in (429, 500, 502, 503, 504):
                resp.raise_for_status()
            return resp

        return with_retry(_do, retry_on=(httpx.TransportError, httpx.HTTPStatusError))

    def close(self) -> None:
        self.client.close()
