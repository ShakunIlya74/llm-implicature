"""Literal listener utilities U(u; s) for quantifiers (G&S-style, N=3).

P_lex(s|u) is uniform over states where u is literally true; U = ln P_lex(s|u).
Incompatible (u, s) get a large negative surrogate so softmax mass ~ 0.
"""

from __future__ import annotations

import math

NEG = -1e9
N_DEFAULT = 3
QUANTS = ("none", "some", "all")


def literal_logprob_state_given_quant(s: int, u: str, n: int = N_DEFAULT) -> float | None:
    """Return ln P_lex(s | u) under uniform literal semantics, or None if impossible."""
    u = u.strip().lower()
    if u not in QUANTS:
        return None
    if not (0 <= s <= n):
        return None
    if u == "none":
        if s != 0:
            return None
        return math.log(1.0)
    if u == "all":
        if s != n:
            return None
        return math.log(1.0)
    # some: true for 1..n inclusive (G&S)
    if u == "some":
        if not (1 <= s <= n):
            return None
        return math.log(1.0 / 3.0)
    return None


def utility_table(s: int, n: int = N_DEFAULT) -> dict[str, float]:
    """U(u; s) = ln P_lex(s|u) with NEG for incompatible pairs."""
    out: dict[str, float] = {}
    for u in QUANTS:
        lp = literal_logprob_state_given_quant(s, u, n=n)
        out[u] = lp if lp is not None else NEG
    return out


def softmax_over_quants(util: dict[str, float], alpha: float) -> dict[str, float]:
    """P(u) ∝ exp(alpha * U(u))."""
    keys = QUANTS
    m = max(alpha * util[k] for k in keys)
    exps = {k: math.exp(alpha * util[k] - m) for k in keys}
    t = sum(exps.values())
    return {k: exps[k] / t for k in keys}
