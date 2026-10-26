"""Signal rules on synthetic price paths."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from mw.signals import rules


def frame(closes, vols=None, eid="US:X"):
    n = len(closes)
    vols = vols or [1_000_000.0] * n
    d0 = date(2025, 1, 1)
    return pl.DataFrame({
        "entity_id": [eid] * n,
        "as_of": [d0 + timedelta(days=i) for i in range(n)],
        "open": [c * 0.995 for c in closes],
        "high": [c * 1.002 for c in closes],
        "low": [c * 0.98 for c in closes],
        "close": closes,
        "volume": vols,
    })  # fmt: skip


def test_breakout_and_spike_detected():
    closes = [100.0] * 80 + [104.0]
    vols = [1e6] * 80 + [3e6]
    s = rules.signals(frame(closes, vols))
    assert s["breakout_buy"][-1]
    closes2 = [10.0] * 30 + [13.0]
    vols2 = [1e6] * 30 + [8e6]
    s2 = rules.signals(frame(closes2, vols2))
    assert s2["volume_spike"][-1]


def test_events_and_score_use_future_only_for_outcome():
    closes = [100.0] * 80 + [104.0] + [110.0] * 12
    vols = [1e6] * 80 + [3e6] + [1e6] * 12
    ev = rules.events(rules.signals(frame(closes, vols)))
    bo = ev.filter(pl.col("signal") == "breakout_buy")
    assert bo.height >= 1 and bo["matured"][0]
    sc = rules.score(bo)
    assert sc["win"][0] and abs(sc["fwd_ret"][0] - (110 / 104 - 1)) < 1e-9


def test_no_signal_on_flat_series():
    s = rules.signals(frame([50.0] * 120))
    assert not any(s[k].any() for k in rules.SIGNALS)
