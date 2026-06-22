"""
aaizaql.core.rate_limiter
─────────────────────────
Token-bucket rate limiter, keyed per tenant/session ID.

Each tenant starts with a full bucket of ``qpm`` tokens that refills at
``qpm / 60`` tokens per second.  One token is consumed per query.  When the
bucket is empty, :exc:`~aaizaql.core.exceptions.RateLimitError` is raised with
a ``retry_after_seconds`` hint.

Set ``qpm=0`` in :class:`~aaizaql.core.config.Settings` to disable rate
limiting entirely.

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import threading
import time
from dataclasses import dataclass, field

from aaizaql.core.exceptions import RateLimitError


@dataclass
class _Bucket:
    """Token-bucket state for a single tenant.

    Attributes:
        tokens: Current token count (float to allow fractional refills).
        last_refill: ``time.monotonic()`` timestamp of the last refill, used to
            compute tokens earned since the previous check.
        lock: Per-bucket mutex so concurrent queries from the same tenant
            do not corrupt the token count.
    """

    tokens: float
    last_refill: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RateLimiter:
    """Token-bucket rate limiter keyed per tenant/session ID.

    Args:
        qpm: Maximum queries per minute per tenant.  Pass ``0`` to disable.

    Example::

        limiter = RateLimiter(qpm=60)
        limiter.check("user-abc")   # consumes one token
    """

    def __init__(self, qpm: int = 60) -> None:
        self._qpm = qpm
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, tenant_id: str) -> None:
        """Consume one token for *tenant_id*, raising if the bucket is empty.

        Refills tokens proportionally to elapsed wall-clock time before checking,
        so the effective rate is ``qpm`` queries per rolling 60-second window.

        Args:
            tenant_id: Opaque string identifying the caller (session UUID, user
                ID, API key, etc.).

        Raises:
            RateLimitError: When the tenant has no tokens remaining.  The error
                carries a ``retry_after_seconds`` hint.
        """
        if self._qpm <= 0:
            return  # rate limiting disabled

        bucket = self._get_bucket(tenant_id)
        with bucket.lock:
            self._refill(bucket)
            if bucket.tokens < 1:
                retry_after = int((1 - bucket.tokens) / (self._qpm / 60.0)) + 1
                raise RateLimitError(tenant_id, retry_after_seconds=retry_after)
            bucket.tokens -= 1

    def _refill(self, bucket: _Bucket) -> None:
        """Add tokens earned since the last refill, capped at ``qpm``.

        Args:
            bucket: The bucket to update in place.  Must be called while the
                caller holds ``bucket.lock``.
        """
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(
            float(self._qpm),
            bucket.tokens + elapsed * (self._qpm / 60.0),
        )
        bucket.last_refill = now

    def _get_bucket(self, tenant_id: str) -> _Bucket:
        """Return the bucket for *tenant_id*, creating it on first access.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            The :class:`_Bucket` for this tenant.
        """
        with self._lock:
            if tenant_id not in self._buckets:
                self._buckets[tenant_id] = _Bucket(tokens=float(self._qpm))
            return self._buckets[tenant_id]
