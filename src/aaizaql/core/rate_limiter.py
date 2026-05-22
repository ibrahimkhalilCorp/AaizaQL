"""
aaizaql.core.rate_limiter
─────────────────────────
T5.3 — Token bucket rate limiter (per tenant_id).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

class RateLimiter:
    """
    Token bucket: each tenant gets `qpm` tokens per minute.
    One token is consumed per query.
    """

    def __init__(self, qpm: int = 60) -> None:
        self._qpm = qpm
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, tenant_id: str) -> None:
        """
        Raise RateLimitError if the tenant has exceeded their quota.
        Otherwise consume one token.
        """
        if self._qpm <= 0:
            return  # disabled
        bucket = self._get_bucket(tenant_id)
        with bucket.lock:
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                self._qpm,
                bucket.tokens + elapsed * (self._qpm / 60.0),
            )
            bucket.last_refill = now
            if bucket.tokens < 1:
                retry_after = int((1 - bucket.tokens) / (self._qpm / 60.0)) + 1
                from aaizaql.core.exceptions import RateLimitError
                raise RateLimitError(tenant_id, retry_after_seconds=retry_after)
            bucket.tokens -= 1

    def _get_bucket(self, tenant_id: str) -> _Bucket:
        with self._lock:
            if tenant_id not in self._buckets:
                self._buckets[tenant_id] = _Bucket(tokens=float(self._qpm))
            return self._buckets[tenant_id]
