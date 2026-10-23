"""Speed-transition detection and scoring on a synthetic speed log shaped like the real ones."""
import numpy as np
import pandas as pd

from grf2com import transitions as tr


def speed_log():
    """Start-up ramp at 0.25 m/s^2, steady 1.25 m/s, two pulses (slow then fast) of 0.5 m/s in 0.43 s edges, stop."""
    rows = [(0.05, 0.0)]
    t, v = 5.0, 0.0
    while v < 1.25 - 1e-9:                       # start-up ramp: 0.01 m/s every 0.04 s
        v = round(v + 0.01, 2)
        rows.append((t, v))
        t += 0.04

    def edge(t0, v0, v1):
        n = int(round(abs(v1 - v0) / 0.04))
        return [(t0 + 0.43 * (k + 1) / n, round(v0 + (v1 - v0) * (k + 1) / n, 3)) for k in range(n)]

    for t0, low in ((60.0, 0.75), (70.0, 1.75)):
        rows += edge(t0, 1.25, low) + edge(t0 + 3.0, low, 1.25)
    rows += edge(100.0, 1.25, 0.0)
    return pd.DataFrame(rows, columns=["time", "speed"])


def test_edges_and_pulses():
    edges = tr.speed_edges(speed_log())
    assert list(edges.kind) == ["start", "change", "change", "change", "change", "stop"]
    ev = tr.speed_events(edges)
    assert list(ev.kind) == ["pulse", "pulse"] and list(ev.direction) == ["slow", "fast"]
    np.testing.assert_allclose(ev.onset, [60.0 + 0.43 / 12, 70.0 + 0.43 / 12], atol=0.01)
    np.testing.assert_allclose(ev.settle, [63.43, 73.43], atol=0.01)
    assert ev.end.iloc[0] == ev.onset.iloc[1]                 # cut at the next pulse
    np.testing.assert_allclose(ev.end.iloc[1], 73.43 + tr.POST_S, atol=0.01)


def test_scores_see_pulse_errors_only_in_windows():
    ev = tr.speed_events(tr.speed_edges(speed_log()))
    n = 100 * 60
    t = np.arange(n) / 60
    true = np.zeros((n, 3))
    pred = np.full((n, 3), 0.005)                             # 5 mm everywhere
    during = np.zeros(n, bool)
    for e in ev.itertuples():
        during |= (t >= e.onset) & (t < e.settle)
    pred[during, 0] = 0.020                                   # 20 mm AP during the pulses
    mask = np.ones(n, bool)
    steady = (15.0, ev.onset.iloc[0] - tr.PRE_S)
    res = tr.score_run({}, {"S_T1": {"pred": pred, "true": true, "mask": mask}}, {"S_T1": (ev, steady)},
                       {"S_T1": 66})
    assert abs(res["steady_rmse_ap_mm"] - 5.0) < 1e-6
    assert res["pulse_rmse_ap_mm"] > 15 and res["pulse_ratio_ap"] > 3
    assert res["trans_rmse_ap_mm"] < res["pulse_rmse_ap_mm"]      # the long window dilutes it
    assert abs(res["pulse_rmse_ml_mm"] - 5.0) < 1e-6
    assert res["n_events"] == 2 and res["recovered_ap_frac"] == 1.0
