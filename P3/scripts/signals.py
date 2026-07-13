"""
Input signal generators for the Lattice Response Spectroscopy experiment (Paper 3, Stage 1).

All signals are scalar time series delta_K(t) of length n_sweeps, meant to be added to the
baseline coupling: K(t) = K0 + delta_K(t), applied uniformly to every bond in the lattice
(spatially uniform driving -- see the protocol document, Section 2).

Every generator is normalized so that the empirical mean input power
    P = mean(delta_K(t)**2)
equals the requested `power` argument as closely as possible. This is the load-bearing
assumption of the whole Stage-1 design: if the powers are not actually equal, any difference
in lattice response could be "explained away" as merely receiving more or less energy, and the
information-vs-energy question collapses. ALWAYS call `check_power()` on the generated signal
before using it and log the result.
"""

import numpy as np


def check_power(signal, target_power, tol=0.05, label=""):
    """Verify empirical power matches target within relative tolerance `tol`. Raises if not."""
    p_emp = float(np.mean(signal ** 2))
    rel_err = abs(p_emp - target_power) / target_power
    status = "OK" if rel_err <= tol else "FAIL"
    print(f"[power check{' ' + label if label else ''}] target={target_power:.6g} "
          f"empirical={p_emp:.6g} rel_err={rel_err:.3%} -> {status}")
    if rel_err > tol:
        raise ValueError(
            f"Signal power mismatch ({label}): target={target_power}, got={p_emp} "
            f"(rel_err={rel_err:.1%} > tol={tol:.1%}). Do not proceed with unequal-power signals."
        )
    return p_emp


def gen_sine(n_sweeps, power, period_sweeps, rng=None, phase0=None):
    """Pure sine wave: delta_K(t) = A*sin(omega*t + phase0), A chosen so mean(delta_K^2)=power.

    period_sweeps: oscillation period in units of MC sweeps (sets omega = 2*pi/period_sweeps).
    """
    A = np.sqrt(2.0 * power)
    omega = 2.0 * np.pi / period_sweeps
    if phase0 is None:
        phase0 = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
    t = np.arange(n_sweeps)
    return A * np.sin(omega * t + phase0)


def gen_phase_flip_sine(n_sweeps, power, period_sweeps, flip_fraction=0.5, rng=None,
                         phase0=None):
    """Counter-driving(파괴적 위상반전) 실험용 신호: flip_fraction 지점까지는 정상적인
    +A*sin(omega*t+phase0)이고, 그 지점부터 끝까지는 부호가 반전된 -A*sin(omega*t+phase0)이다.

    flip_fraction=1.0이면 뒤집는 지점이 배열 끝이라 사실상 뒤집기가 없는 대조군(control)이
    된다 -- flip timing sweep에서 0.25/0.5/0.75와 나란히 비교하기 위한 기준점.

    파워는 flip 여부와 무관하게 항상 정확히 A^2/2=power로 고정된다 (부호만 바뀌므로).
    """
    A = np.sqrt(2.0 * power)
    omega = 2.0 * np.pi / period_sweeps
    if phase0 is None:
        phase0 = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
    t = np.arange(n_sweeps)
    base = A * np.sin(omega * t + phase0)
    flip_point = int(round(n_sweeps * flip_fraction))
    sign = np.ones(n_sweeps)
    sign[flip_point:] = -1.0
    return base * sign


def gen_square_wave(n_sweeps, power, period_sweeps, rng=None, phase0=None):
    """결정론적 사각파: delta_K(t) = +-A, period_sweeps 스윕마다 부호가 바뀐다.

    dwell_time_sweep.py에서 발견한 대로, gen_random_telegraph(dwell_sweeps=1)은 사실
    "랜덤 과정"이 아니라 이 사각파의 period_sweeps=2인 특수한 경우와 동일하다 (매 스윕마다
    무조건 부호가 바뀌므로 확률적 요소가 전혀 없음). 이 함수는 그 사각파 계열을
    period_sweeps를 자유롭게 바꿀 수 있는 독립된 신호로 명시적으로 분리한 것이다 --
    random telegraph(dwell을 확률적으로 뽑음)와 혼동하지 않도록, 별도의 이름/함수로 둔다.

    power = A^2 이므로(항상 +-A이므로 amplitude^2가 그대로 power), dwell_sweeps나
    period_sweeps에 관계없이 파워가 자동으로 정확히 고정된다 -- random telegraph와
    마찬가지로 파워 정규화가 거의 공짜로 이루어진다.
    """
    A = np.sqrt(power)
    if phase0 is None:
        phase0 = int(rng.integers(0, period_sweeps)) if rng is not None else 0
    t = np.arange(n_sweeps)
    half = period_sweeps / 2.0
    phase_t = (t + phase0) % period_sweeps
    return np.where(phase_t < half, A, -A)


def gen_white_noise(n_sweeps, power, rng):
    """White noise: i.i.d. Gaussian at every sweep, flat power spectrum."""
    sigma = np.sqrt(power)
    signal = rng.normal(0.0, sigma, size=n_sweeps)
    # Rescale to hit the target power exactly (finite-sample variance will not be exact).
    signal *= np.sqrt(power / np.mean(signal ** 2))
    return signal


def gen_random_telegraph(n_sweeps, power, dwell_sweeps, rng):
    """Random telegraph signal: delta_K(t) = +-A, holding each sign for a random dwell time
    (geometrically distributed with mean `dwell_sweeps`) before flipping.

    Power is exactly A^2 regardless of dwell time (the signal is always +-A) -- only the
    spectral shape (Lorentzian, corner frequency ~1/dwell_sweeps) changes with dwell time,
    which is exactly the point: same power and similar coarse "randomness" as white noise,
    but a different, tunable, temporal correlation structure.
    """
    A = np.sqrt(power)
    signal = np.empty(n_sweeps)
    t = 0
    sign = rng.choice([-1.0, 1.0])
    p_flip = 1.0 / dwell_sweeps  # geometric distribution: dwell ~ Geometric(p_flip)
    while t < n_sweeps:
        dwell = rng.geometric(p_flip)
        dwell = min(dwell, n_sweeps - t)
        signal[t:t + dwell] = sign * A
        sign *= -1.0
        t += dwell
    return signal


def shannon_entropy_of_signal(signal, n_bins=16):
    """Histogram-based Shannon entropy (in bits) of the signal's amplitude distribution.

    This is a simple, model-free way to quantify "how much information" a given input carries
    per sample -- e.g. a pure sine wave spends most of its time near +-A (bimodal, low entropy
    given a fixed variance), white noise is close to a Gaussian, and the telegraph signal is
    the extreme two-point case. Used for the Section 6 "input entropy vs output" comparison.
    """
    counts, _ = np.histogram(signal, bins=n_bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


SIGNAL_REGISTRY = {
    "sine": gen_sine,
    "white": gen_white_noise,
    "telegraph": gen_random_telegraph,
    "square": gen_square_wave,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_sweeps = 20000
    power = 0.01  # (delta_K)^2, keep small relative to K0 for the linear-response regime

    sine = gen_sine(n_sweeps, power, period_sweeps=50, rng=rng)
    white = gen_white_noise(n_sweeps, power, rng=rng)
    teleg = gen_random_telegraph(n_sweeps, power, dwell_sweeps=25, rng=rng)

    for name, sig in [("sine", sine), ("white", white), ("telegraph", teleg)]:
        check_power(sig, power, label=name)
        h = shannon_entropy_of_signal(sig)
        print(f"  {name}: Shannon entropy = {h:.3f} bits\n")
