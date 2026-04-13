"""429-aware adaptive concurrency for batched HTTP/API calls.

Uses a pool of at most ``max_concurrency`` worker threads; an internal limit
(starting at ``initial``) caps how many requests may be in flight. On HTTP 429 /
rate-limit errors the limit is reduced; after enough consecutive successes it
increases (up to ``max_concurrency``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection of rate limits (OpenRouter / OpenAI SDK)."""
    try:
        import openai

        if isinstance(exc, getattr(openai, "RateLimitError", ())):
            return True
        if isinstance(exc, getattr(openai, "APIStatusError", ())):
            return getattr(exc, "status_code", None) == 429
    except Exception:
        pass
    sc = getattr(exc, "status_code", None)
    if sc == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


@dataclass
class DynamicConcurrencyConfig:
    initial: int = 15
    min_concurrency: int = 2
    max_concurrency: int = 64
    step_up: int = 2
    """Raise limit by this much after ``success_streak_to_ramp`` consecutive successes."""
    success_streak_to_ramp: int = 30
    """On 429, new_limit = max(min, floor(limit * down429_ratio))."""
    down429_ratio: float = 0.5
    retry_base_sleep_s: float = 1.0
    retry_max_sleep_s: float = 60.0
    max_retries_429: int = 100
    """Stop after this many rate-limit retries per task (then re-raise last error)."""


@dataclass
class _Controller:
    cfg: DynamicConcurrencyConfig
    limit: int = field(init=False)
    in_flight: int = 0
    success_streak: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(init=False)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        lo, hi = self.cfg.min_concurrency, self.cfg.max_concurrency
        start = self.cfg.initial
        self.limit = min(hi, max(lo, start))
        self.cond = threading.Condition(self.lock)

    def _log(self, kind: str, **extra: Any) -> None:
        self.events.append(
            {"t": time.time(), "kind": kind, "limit": self.limit, **extra}
        )
        if len(self.events) > 800:
            self.events = self.events[-500:]

    def acquire_slot(self) -> None:
        with self.cond:
            while self.in_flight >= self.limit:
                self.cond.wait()
            self.in_flight += 1

    def release_slot(self) -> None:
        with self.cond:
            self.in_flight -= 1
            self.cond.notify_all()

    def on_success(self) -> None:
        with self.cond:
            self.success_streak += 1
            if (
                self.success_streak >= self.cfg.success_streak_to_ramp
                and self.limit < self.cfg.max_concurrency
            ):
                old = self.limit
                self.limit = min(
                    self.cfg.max_concurrency, self.limit + self.cfg.step_up
                )
                self.success_streak = 0
                self._log("ramp_up", old_limit=old, new_limit=self.limit)
            self.cond.notify_all()

    def on_429(self) -> None:
        with self.cond:
            old = self.limit
            new_lim = max(
                self.cfg.min_concurrency,
                int(old * self.cfg.down429_ratio),
            )
            if new_lim < old:
                self.limit = new_lim
                self._log("ramp_down_429", old_limit=old, new_limit=self.limit)
            self.success_streak = 0
            self.cond.notify_all()


def run_adaptive_map(
    payloads: Sequence[T],
    fn: Callable[[T], R],
    cfg: DynamicConcurrencyConfig | None = None,
) -> tuple[list[R], dict[str, Any]]:
    """Run ``fn`` over payloads with adaptive concurrency.

    Returns (results in input order, stats dict with events tail and final limit).
    """
    cfg = cfg or DynamicConcurrencyConfig()
    ctrl = _Controller(cfg=cfg)
    n = len(payloads)
    results: list[Any] = [None] * n

    def worker(index: int, payload: T) -> None:
        sleep_s = cfg.retry_base_sleep_s
        rl_count = 0
        while True:
            ctrl.acquire_slot()
            try:
                results[index] = fn(payload)
                ctrl.on_success()
                return
            except BaseException as e:
                if is_rate_limit_error(e):
                    ctrl.on_429()
                    rl_count += 1
                    if rl_count > cfg.max_retries_429:
                        raise
                else:
                    raise
            finally:
                ctrl.release_slot()
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 2.0, cfg.retry_max_sleep_s)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as ex:
        futures = [ex.submit(worker, i, payloads[i]) for i in range(n)]
        for fut in as_completed(futures):
            fut.result()

    stats = {
        "final_limit": ctrl.limit,
        "event_count": len(ctrl.events),
        "events_tail": ctrl.events[-80:],
    }
    return results, stats  # type: ignore[return-value]
