import numpy as np

from mw.research import setup_sim, setups
from mw.research.setups import Armed


def test_pivots_are_confirmed_only_after_the_move_away() -> None:
    # up to 110 at bar 10, then down; the high is known only once price is 4%+ below it
    up = np.linspace(100, 110, 11)
    down = np.linspace(109, 95, 15)
    h = np.r_[up, down] + 0.5
    lo = np.r_[up, down] - 0.5
    atr = np.full(len(h), 0.5)
    pv = setups.pivots(h, lo, atr)
    i = int(np.flatnonzero(pv.kind == 1)[0])
    assert pv.idx[i] == 10
    assert pv.confirmed[i] > pv.idx[i]
    # confirmation bar is the first bar whose low is >= 4% below the high (110.5 * 0.96 = 106.08)
    assert (
        lo[pv.confirmed[i]] <= 110.5 - 0.04 * 110.5
        and lo[pv.confirmed[i] - 1] > 110.5 - 0.04 * 110.5
    )


def _bars(
    o: list[float], h: list[float], lo: list[float], c: list[float]
) -> tuple[np.ndarray, ...]:
    return tuple(np.array(x, float) for x in (o, h, lo, c))


def test_buy_stop_fills_at_level_or_open_and_skips_extended_gaps() -> None:
    o, h, lo, c = _bars(
        [10, 10, 10.2, 10], [10, 10.1, 10.6, 10.3], [9.9, 9.9, 10.1, 9.9], [10, 10, 10.5, 10.1]
    )
    a = Armed("x", 0, level=10.3, stop=9.5, target=None, valid=3)
    assert setup_sim.entry(a, o, h, c) == (2, 10.3)  # day 2 high reaches 10.3: filled at the level
    o2 = o.copy()
    o2[2] = 10.9  # opens > 5% above the level: do not chase
    assert setup_sim.entry(a, o2, h, c) is None
    c2 = c.copy()
    c2[1] = 9.4  # closes below the stop before triggering: the pattern failed
    assert setup_sim.entry(a, o, h, c2) is None


def test_manage_stop_and_target() -> None:
    n = 10
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    lo = np.full(n, 99.0)
    c = np.full(n, 100.0)
    ma = np.full(n, 90.0)
    lo[3] = 94.0  # touches the stop (95)
    r, held, why = setup_sim.manage("target", 0, 100.0, 95.0, 110.0, o, h, lo, c, ma, ma)
    assert r == -1.0 and held == 4 and why == "損切り"
    lo[3] = 99.0
    h[5] = 111.0  # reaches the target 110 -> +2R
    r, held, why = setup_sim.manage("target", 0, 100.0, 95.0, 110.0, o, h, lo, c, ma, ma)
    assert r == 2.0 and why == "利確"
