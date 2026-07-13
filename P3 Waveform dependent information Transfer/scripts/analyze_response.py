"""
Post-processing analysis for the Lattice Response Spectroscopy experiment (Paper 3, Stage 1).

Implements the Section-6 analyses, revised after a pilot-stage review (see the reliability
tiering at the bottom of this docstring):
  1. Cross-correlation response time tau(input -> output) -- PAIRWISE-VALID, not interpolated.
  2. Transfer entropy TE(input -> output) -- also pairwise-valid.
  3. Phase lag phi(omega) for a periodic (sine) drive -- replaces cross-correlation tau for
     that case, since argmax(cross-correlation) against a periodic input is itself periodic
     in lag and does not define a single response time (confirmed empirically: the same run
     reported tau=14 at max_lag=50 and tau=101 at max_lag=150 -- see recompute_tau.py).
  4. Efficiency (eta = output / absorbed energy) and input Shannon entropy.

On NaN handling (drop is undefined -- not just "missing" -- at any measurement step with no
vortex present): earlier versions of this module linearly interpolated over NaNs before
computing statistics. That is no longer done. drop's domain of definition is genuinely
"vortex present"; manufacturing a value where the physical quantity does not exist is not
appropriate for a result meant to go in a paper. Every function below instead uses PAIRWISE
VALID samples -- for a given lag or a given (y_t, y_t-1, x_t-1) triple, only time points
where every required quantity is finite are used, and the sample count actually used is
reported so a thin, unreliable estimate is visible rather than silently accepted.

Reliability tiers (for reporting in the paper; see protocol document Section 6):
  - High confidence:   absorbed energy, birth efficiency, net TE(input->birth),
                        tau(input->output) for White / Random-telegraph (empirically stable
                        across search-window size).
  - Conditional:       TE(input->drop) / correlations involving drop (pairwise-valid count
                        should be reported alongside the estimate), TE for the Sine case.
  - Method change:     tau for the Sine case -- report phase lag phi(omega) instead of a
                        cross-correlation argmax.
"""

import numpy as np


MIN_VALID_FRACTION = 0.3  # below this fraction of jointly-valid samples, refuse to report


# ------------------------------------------------------------------------------
# 1. Cross-correlation response time (pairwise-valid)
# ------------------------------------------------------------------------------

def cross_correlation_lag(x, y, max_lag, min_valid=30):
    """Normalized cross-correlation r(lag) = corr(x(t), y(t+lag)) for lag in [-max_lag, max_lag],
    using only pairs where BOTH x(t) and y(t+lag) are finite (pairwise-valid deletion -- no
    values are invented for lags/points where a quantity like `drop` is undefined).

    Returns (lags, r_values, n_valid_per_lag, tau_best, r_best, n_valid_best). If fewer than
    `min_valid` pairs are available at the best lag, r_best is NaN and tau_best is None,
    signalling "cannot be estimated" rather than returning a number computed from too few
    points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.full(len(lags), np.nan)
    n_valid_arr = np.zeros(len(lags), dtype=int)

    for k, lag in enumerate(lags):
        if lag >= 0:
            xs, ys = x[:n - lag], y[lag:]
        else:
            xs, ys = x[-lag:], y[:n + lag]
        mask = np.isfinite(xs) & np.isfinite(ys)
        n_valid = int(mask.sum())
        n_valid_arr[k] = n_valid
        if n_valid < min_valid:
            continue
        xv, yv = xs[mask], ys[mask]
        xv = (xv - xv.mean()) / (xv.std() + 1e-15)
        yv = (yv - yv.mean()) / (yv.std() + 1e-15)
        r[k] = np.mean(xv * yv)

    if np.all(np.isnan(r)):
        return lags, r, n_valid_arr, None, float("nan"), 0

    best = int(np.nanargmax(np.abs(r)))
    return lags, r, n_valid_arr, int(lags[best]), float(r[best]), int(n_valid_arr[best])


# ------------------------------------------------------------------------------
# 2. Transfer entropy (discrete-bin estimator, pairwise-valid)
# ------------------------------------------------------------------------------

def _discretize(x, n_bins):
    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    edges[-1] += 1e-9
    return np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)


def _entropy(labels, n_states):
    counts = np.bincount(labels, minlength=n_states)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _joint_entropy(label_tuples, n_states_tuple):
    mult = np.ones(len(label_tuples), dtype=np.int64)
    for i in range(1, len(label_tuples)):
        mult[i] = mult[i - 1] * n_states_tuple[i - 1]
    joint = np.zeros(len(label_tuples[0]), dtype=np.int64)
    for lab, m in zip(label_tuples, mult):
        joint += lab * m
    return _entropy(joint, int(np.prod(n_states_tuple)))


def transfer_entropy(source, target, n_bins=6, lag=1, min_valid=50):
    """TE(source -> target) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag}), discrete-bin
    plug-in estimator (matches the Paper 2 causal-analysis methodology: 6 bins, lag=1).

    Pairwise-valid: only time points where target[t], target[t-lag], AND source[t-lag] are
    ALL finite are used (a NaN in `drop` at any one of those three positions excludes that
    time point, rather than being interpolated). Returns (TE, n_valid). If n_valid < min_valid,
    TE is returned as NaN.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(source)
    assert len(target) == n

    y_t_raw = target[lag:]
    y_tm1_raw = target[:-lag]
    x_tm1_raw = source[:-lag]
    joint_valid = np.isfinite(y_t_raw) & np.isfinite(y_tm1_raw) & np.isfinite(x_tm1_raw)
    n_valid = int(joint_valid.sum())
    if n_valid < min_valid:
        return float("nan"), n_valid

    y_t_raw = y_t_raw[joint_valid]
    y_tm1_raw = y_tm1_raw[joint_valid]
    x_tm1_raw = x_tm1_raw[joint_valid]

    # bin edges from the valid subset only (consistent with what's actually being used)
    x = _discretize(x_tm1_raw, n_bins)
    y_t = _discretize(y_t_raw, n_bins)
    y_tm1 = _discretize(y_tm1_raw, n_bins)

    H_y_ytm1 = _joint_entropy([y_t, y_tm1], (n_bins, n_bins))
    H_ytm1 = _entropy(y_tm1, n_bins)
    H_cond_y = H_y_ytm1 - H_ytm1

    H_y_ytm1_xtm1 = _joint_entropy([y_t, y_tm1, x], (n_bins, n_bins, n_bins))
    H_ytm1_xtm1 = _joint_entropy([y_tm1, x], (n_bins, n_bins))
    H_cond_yx = H_y_ytm1_xtm1 - H_ytm1_xtm1

    return float(H_cond_y - H_cond_yx), n_valid


def net_transfer_entropy(x, y, n_bins=6, lag=1, min_valid=50):
    """Net TE = TE(x->y) - TE(y->x). Positive means x drives y more than the reverse.
    Returns (net_TE, TE_xy, TE_yx, n_valid_xy, n_valid_yx).
    """
    te_xy, n_xy = transfer_entropy(x, y, n_bins, lag, min_valid)
    te_yx, n_yx = transfer_entropy(y, x, n_bins, lag, min_valid)
    net = te_xy - te_yx if (np.isfinite(te_xy) and np.isfinite(te_yx)) else float("nan")
    return net, te_xy, te_yx, n_xy, n_yx


# ------------------------------------------------------------------------------
# 3. Phase lag at a known drive frequency (for periodic / Sine inputs)
# ------------------------------------------------------------------------------

def phase_lag_at_frequency(x, y, omega, min_valid=30):
    """Single-frequency (lock-in-amplifier-style) phase lag between a periodic input x(t) and
    an output y(t), at known angular frequency `omega` (radians per sample, in the SAME time
    units as x, y -- i.e. per measurement step, not per raw sweep).

    Method: project both series onto exp(-i*omega*t); the phase of each projection is that
    series' phase at frequency omega, and the difference is the response's phase lag,
    independent of whatever phase offset the input itself started at. This is well-defined
    for ANY output amplitude (unlike cross-correlation argmax), and does not suffer the
    periodic-aliasing ambiguity that makes a single cross-correlation-based tau meaningless
    for a periodic drive.

    Pairwise-valid: time points where y(t) is not finite (e.g. `drop` with no vortex present)
    are excluded from both projections.

    Returns (phi_rad, amp_x, amp_y, n_valid). phi_rad is wrapped to (-pi, pi]; phi_rad > 0
    means y LAGS x (y peaks after x, in time). If n_valid < min_valid, phi_rad is NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.arange(len(x))
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        return float("nan"), float("nan"), float("nan"), n_valid

    tv, xv, yv = t[valid], x[valid], y[valid]
    basis = np.exp(-1j * omega * tv)
    X = np.sum(xv * basis)
    Y = np.sum(yv * basis)
    phase_x = np.angle(X)
    phase_y = np.angle(Y)
    phi = phase_y - phase_x
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    amp_x = 2.0 * np.abs(X) / n_valid
    amp_y = 2.0 * np.abs(Y) / n_valid
    return float(phi), float(amp_x), float(amp_y), n_valid


def omega_from_period(period_sweeps, meas_stride):
    """Angular frequency in radians per MEASUREMENT STEP (not per raw sweep), matching the
    sample spacing of the saved time series (which are recorded every meas_stride sweeps).
    """
    period_in_meas_steps = period_sweeps / meas_stride
    return 2.0 * np.pi / period_in_meas_steps


# ------------------------------------------------------------------------------
# TE 신뢰도 평가: 3개 지표 x 3단계 (FAIL/WARNING/PASS) -- 채찍GPT 피드백 반영
# ------------------------------------------------------------------------------
#
# 처음에는 "period/stride >= 8" 하나만으로 이분법(reliable/unreliable) 판정을 했는데,
# 이건 두 가지를 놓친다: (1) 경계 근처의 애매한 경우를 그냥 통과/실패로만 나누는 것보다
# 3단계(FAIL/WARNING/PASS)가 더 정직하고, (2) 한 주기를 잘 샘플링해도 전체 실행 시간
# 안에 주기가 몇 번이나 반복되는지(통계적으로 충분한 반복이 있는지)는 별개의 문제다 --
# 예를 들어 총 35000 sweep에 period=2048이면 겨우 17주기뿐이라 TE 추정 자체가 흔들릴 수
# 있다. 그래서 이제 세 가지를 따로 본다:
#   1. Samples per period  (=timescale/meas_stride)  -- 한 주기를 얼마나 세밀하게 쟀는가
#   2. Number of cycles     (=total_sweeps/timescale) -- 그 주기가 몇 번이나 반복됐는가
#   3. tau_signal/meas_stride (telegraph류 전용)      -- 실측 상관시간이 stride보다 느린가

SAMPLES_PER_PERIOD_THRESHOLDS = (5.0, 10.0)   # (FAIL 미만, WARNING 미만 -- 그 이상 PASS)
N_CYCLES_THRESHOLDS = (20.0, 50.0)             # 위와 동일한 해석


def _tier(value, fail_below, warn_below):
    if value < fail_below:
        return "FAIL"
    elif value < warn_below:
        return "WARNING"
    else:
        return "PASS"


def _combine_tiers(*tiers):
    """여러 단계 중 가장 나쁜 것을 전체 등급으로 -- 하나라도 FAIL이면 LOW, 하나라도
    WARNING이면(FAIL 없이) MEDIUM, 전부 PASS면 HIGH.
    """
    if "FAIL" in tiers:
        return "LOW"
    elif "WARNING" in tiers:
        return "MEDIUM"
    else:
        return "HIGH"


# ------------------------------------------------------------------------------
# Effect size (baseline vs driven) -- 채찍GPT 권고: p<0.05뿐 아니라 "효과가 실제로
# 얼마나 큰가"도 같이 보고해야 리뷰어가 신뢰한다.
# ------------------------------------------------------------------------------

def cohens_d(baseline, driven, hedges_correction=True):
    """평형(baseline) 구간과 구동(driven) 구간의 차이를 표준편차 단위로 나타낸 효과크기.

    d = (mean(driven) - mean(baseline)) / pooled_std

    hedges_correction=True(기본값)이면 작은 표본에서의 편향을 보정한 Hedges' g를
    반환한다 (n이 크면 d와 거의 같아진다). NaN은 무시하고 계산한다 (pairwise-valid).

    관례적 해석 기준(Cohen 1988): |d|<0.2 매우 작음, 0.2~0.5 작음, 0.5~0.8 중간,
    >0.8 큼. 이 기준은 안내용일 뿐이며 분야/맥락에 따라 다르게 해석될 수 있다.

    Returns (d_or_g, n_baseline_valid, n_driven_valid).
    """
    b = np.asarray(baseline, dtype=float)
    d_arr = np.asarray(driven, dtype=float)
    b = b[np.isfinite(b)]
    d_arr = d_arr[np.isfinite(d_arr)]
    nb, nd = len(b), len(d_arr)
    if nb < 2 or nd < 2:
        return float("nan"), nb, nd

    mean_b, mean_d = b.mean(), d_arr.mean()
    var_b, var_d = b.var(ddof=1), d_arr.var(ddof=1)
    pooled_var = ((nb - 1) * var_b + (nd - 1) * var_d) / (nb + nd - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std < 1e-15:
        return float("nan"), nb, nd

    d = (mean_d - mean_b) / pooled_std
    if hedges_correction:
        # Hedges' g 보정항 (작은 표본 편향 보정, 표본이 크면 1에 가까워짐)
        n_total = nb + nd
        correction = 1.0 - 3.0 / (4.0 * n_total - 9.0)
        d = d * correction
    return d, nb, nd


def effect_size_label(d):
    """Cohen(1988)의 관례적 해석 기준 -- 안내용."""
    if not np.isfinite(d):
        return "?"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"



def assess_te_reliability(signal_timescale_sweeps, meas_stride, total_sweeps,
                           tau_signal=None):
    """TE 신뢰도를 3개 지표로 평가한다 (samples/period, n_cycles, 있으면 tau_signal도).

    signal_timescale_sweeps=inf (white noise처럼 뚜렷한 주기가 없는 신호의 관례적 값)이면
    이 체크 자체가 적용 대상이 아니므로 곧바로 overall="HIGH"를 반환한다 -- inf로 나누면
    n_cycles가 0이 되어 버려서(=FAIL) 잘못 걸리는 문제를 여기서 막는다.

    Returns a dict:
      samples_per_period, samples_per_period_tier,
      n_cycles, n_cycles_tier,
      tau_ratio, tau_tier (tau_signal이 주어진 경우만),
      overall ("HIGH"/"MEDIUM"/"LOW")
    """
    if not np.isfinite(signal_timescale_sweeps):
        return {
            "samples_per_period": float("inf"), "samples_per_period_tier": "PASS",
            "n_cycles": float("inf"), "n_cycles_tier": "PASS",
            "tau_ratio": None, "tau_tier": None, "overall": "HIGH",
        }

    samples_per_period = signal_timescale_sweeps / meas_stride
    spp_tier = _tier(samples_per_period, *SAMPLES_PER_PERIOD_THRESHOLDS)

    n_cycles = total_sweeps / signal_timescale_sweeps
    cycles_tier = _tier(n_cycles, *N_CYCLES_THRESHOLDS)

    tiers_for_overall = [spp_tier, cycles_tier]
    result = {
        "samples_per_period": samples_per_period,
        "samples_per_period_tier": spp_tier,
        "n_cycles": n_cycles,
        "n_cycles_tier": cycles_tier,
        "tau_ratio": None,
        "tau_tier": None,
    }
    if tau_signal is not None:
        tau_ratio = tau_signal / meas_stride
        tau_tier = "WARNING" if tau_ratio < 1.0 else "PASS"
        result["tau_ratio"] = tau_ratio
        result["tau_tier"] = tau_tier
        tiers_for_overall.append(tau_tier)

    result["overall"] = _combine_tiers(*tiers_for_overall)
    return result


# 이전 버전과의 호환을 위해 남겨둠 (summarize_case 등에서 여전히 단순 bool이 필요한 곳).
TE_MIN_PERIOD_TO_STRIDE_RATIO = SAMPLES_PER_PERIOD_THRESHOLDS[1]  # = 10.0 (PASS 기준)


def check_nyquist_ratio(signal_timescale_sweeps, meas_stride, min_ratio=TE_MIN_PERIOD_TO_STRIDE_RATIO):
    """(레거시) 단순 이분법 버전 -- 새 코드는 assess_te_reliability()를 쓰세요.
    Returns (ratio, is_reliable) where is_reliable = (ratio >= min_ratio), 즉 3단계 중
    'PASS'에 해당하는 것만 True로 취급한다 (WARNING도 False로 묶임 -- 더 엄격한 이분법).
    """
    ratio = signal_timescale_sweeps / meas_stride
    return ratio, ratio >= min_ratio


# 4. Efficiency and input entropy
# ------------------------------------------------------------------------------

def absorbed_energy(energy_driven, energy_baseline_mean):
    return energy_driven - energy_baseline_mean


def efficiency(output_driven, output_baseline_mean, energy_driven, energy_baseline_mean,
               eps=1e-12):
    """eta(t) = Delta(output) / Delta(absorbed energy). NaN in output_driven (e.g. drop with
    no vortex) propagates to NaN in eta(t) at that step, rather than being filled -- take
    np.nanmean() over the returned series for a scalar summary.
    """
    d_out = output_driven - output_baseline_mean
    d_E = absorbed_energy(energy_driven, energy_baseline_mean)
    d_E_safe = np.where(np.abs(d_E) < eps, np.nan, d_E)
    return d_out / d_E_safe


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 5000

    # --- cross-correlation + TE self-test (unchanged from before, plus NaN robustness) ---
    true_lag = 4
    x = rng.normal(size=n)
    y = np.empty(n)
    y[:true_lag] = rng.normal(size=true_lag)
    y[true_lag:] = 0.8 * x[:-true_lag] + 0.3 * rng.normal(size=n - true_lag)

    lags, r, n_valid_arr, tau_best, r_best, n_valid_best = cross_correlation_lag(x, y, max_lag=20)
    print(f"Cross-correlation self-test: true_lag={true_lag}, recovered tau={tau_best}, "
          f"r={r_best:.3f}, n_valid={n_valid_best} (expect tau close to {true_lag})")
    assert tau_best == true_lag

    # inject NaN like `drop` would have, and confirm pairwise-valid still recovers the lag
    y_nan = y.copy()
    y_nan[rng.random(n) < 0.3] = np.nan
    lags, r, n_valid_arr, tau2, r2, nv2 = cross_correlation_lag(x, y_nan, max_lag=20)
    print(f"Same test with 30% NaN injected: tau={tau2}, r={r2:.3f}, n_valid={nv2} "
          f"(should still recover tau={true_lag}, with n_valid roughly 70% of clean case)")
    assert tau2 == true_lag

    net_te, te_xy, te_yx, n_xy, n_yx = net_transfer_entropy(x, y, n_bins=6, lag=1)
    print(f"Transfer entropy self-test: TE(x->y)={te_xy:.4f} (n={n_xy}), "
          f"TE(y->x)={te_yx:.4f} (n={n_yx}), net={net_te:.4f}")
    assert te_xy > te_yx

    # --- phase-lag self-test: known phase shift at a known frequency ---
    true_phi = 0.7  # radians
    omega = 2 * np.pi / 37.0  # arbitrary non-integer period, in samples
    t = np.arange(n)
    drive = np.sin(omega * t + 1.3)  # arbitrary input phase offset -- should not matter
    response = np.sin(omega * t + 1.3 + true_phi) + 0.1 * rng.normal(size=n)
    phi_est, amp_x, amp_y, nv = phase_lag_at_frequency(drive, response, omega)
    print(f"\nPhase-lag self-test: true_phi={true_phi:.3f} rad, recovered={phi_est:.3f} rad "
          f"(amp_x={amp_x:.3f}, amp_y={amp_y:.3f}, n_valid={nv})")
    assert abs(phi_est - true_phi) < 0.05, "phase-lag estimate should be accurate to <0.05 rad"

    # and with NaN gaps (simulating drop-with-no-vortex dropouts)
    response_nan = response.copy()
    response_nan[rng.random(n) < 0.4] = np.nan
    phi_est2, _, _, nv2 = phase_lag_at_frequency(drive, response_nan, omega)
    print(f"Same test with 40% NaN: recovered phi={phi_est2:.3f} rad, n_valid={nv2} "
          f"(should still be close to true_phi={true_phi:.3f})")
    assert abs(phi_est2 - true_phi) < 0.1

    print("\nAll self-tests passed (pairwise-valid cross-correlation, transfer entropy, "
          "and phase-lag estimation all confirmed NaN-robust without inventing values).")
