"""Replay of the lab's Song et al. (2020) filter on an idealized gait: walking in place on a 1.2 m/s belt."""
import numpy as np

from grf2com.baselines.song_kf import LOOP_DT, calibrated_mass, run_filter

FS, STEP_S, STANCE_S, SPEED, LAND_AT, MASS = 1200, 0.55, 0.65, 1.2, 0.9, 70.0


def ideal_gait(duration_s=30.0, start_s=0.0, stand_weight_n=0.0):
    t = np.arange(int(duration_s * FS)) / FS
    sig = {"t": t, "Fy_l": np.zeros_like(t), "Fy_r": np.zeros_like(t), "speed": np.where(t >= start_s, SPEED, 0.0)}
    for k, side in enumerate("lr"):
        fz = np.where(t < start_s, stand_weight_n / 2, 0.0)
        cop = np.full_like(t, LAND_AT)
        for c in np.arange(start_s + 1.0 + k * STEP_S, duration_s - STANCE_S, 2 * STEP_S):
            on = (t >= c) & (t < c + STANCE_S)
            fz[on] = 600.0
            cop[on] = LAND_AT - SPEED * (t[on] - c)          # the planted foot moves back with the belt
        sig[f"Fz_{side}"], sig[f"COPy_{side}"] = fz, cop
    return sig


def test_measurements_and_convergence():
    res = run_filter(ideal_gait(), MASS, FS)
    st = res["steps"].iloc[2:]                              # skip the start-up steps
    assert st.used.all()
    assert np.allclose(st.step_t, STEP_S, atol=LOOP_DT + 1e-9)
    trailing = LAND_AT - SPEED * STEP_S                      # previous foot carried back over one step
    mid = (LAND_AT + trailing) / 2 - SPEED * LOOP_DT         # the code shifts both feet once more on the step tick
    assert np.all(np.abs(st.com_mes - mid) < 0.03)
    assert abs(st.dcom_mes.mean()) < 0.01 and np.all(np.abs(st.dcom_mes) < 0.05)
    late = res["t"] > 20
    assert np.all(np.abs(res["p"][late] - mid) < 0.03) and np.all(np.abs(res["v"][late]) < 0.05)
    assert np.allclose(np.diff(res["t"]), LOOP_DT)


def test_causal_start_and_mass():
    a = run_filter(ideal_gait(), MASS, FS)
    sig = ideal_gait()
    cut = 15.0
    sig["Fy_l"][sig["t"] > cut] += 50.0                       # change only the future
    b = run_filter(sig, MASS, FS)
    past = a["t"] <= cut - LOOP_DT
    np.testing.assert_allclose(a["p"][past], b["p"][past])
    # The filter starts with the treadmill; mass comes from standing before that.
    sig = ideal_gait(start_s=3.0, stand_weight_n=75.0 * 9.81)
    res = run_filter(sig, MASS, FS)
    assert res["t"][0] >= 3.0
    assert abs(calibrated_mass(sig, MASS) - 75.0) < 0.1


def test_streaming_filter_matches_replay():
    from grf2com.baselines.song_kf import LabKalmanStream

    sig = ideal_gait(start_s=2.0)
    sig["Fy_l"] = sig["Fy_l"] + 30.0 * np.sin(sig["t"] * 7.0)        # some force input to integrate
    ref = run_filter(sig, MASS, FS)
    i0 = int(np.flatnonzero(sig["speed"] > 0)[0])
    t_start = sig["t"][i0]
    rows = np.minimum(np.searchsorted(sig["t"], ref["t"], side="right") - 1, len(sig["t"]) - 1)
    kf = LabKalmanStream(MASS, skip_repeated=False)
    p, steps = [], []
    for k, r in enumerate(rows):
        st = kf.tick(ref["t"][k] - t_start, sig["Fy_l"][r] + sig["Fy_r"][r], sig["Fz_l"][r], sig["Fz_r"][r],
                     sig["COPy_l"][r], sig["COPy_r"][r], sig["speed"][r])
        p.append(kf.x0)
        if st:
            steps.append(st)
    np.testing.assert_allclose(p, ref["p"], rtol=0, atol=1e-12)
    assert len(steps) == len(ref["steps"])
    np.testing.assert_allclose([s["com_step"] for s in steps], ref["steps"].com_step, atol=1e-12)
    assert [s["used"] for s in steps] == ref["steps"].used.tolist()
