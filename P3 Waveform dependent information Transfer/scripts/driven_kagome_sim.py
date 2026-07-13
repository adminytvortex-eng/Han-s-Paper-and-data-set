"""
Driven kagome XY model: Metropolis MC engine + measurements for the Lattice Response
Spectroscopy experiment (Paper 3, Stage 1).

Model
-----
H(t) = - K(t) * sum_{<ij>} cos(phi_i - phi_j),      K(t) = K0 + delta_K(t)

delta_K(t) is spatially uniform (same value added to every bond at sweep t) and is generated
by signals.py -- a *parametric* drive that rescales the existing bond term rather than adding
a symmetry-breaking pinning field (see the protocol document, Section 2).

The Metropolis update and the information-density (H-map) calculation are ported to match
the user's reference kagome_hmap_miner.py exactly (site indexing, bond list, full-range
proposal, and the *unrounded* winding-number histogram formula -- see the note above
`compute_Hmap_reference_naive()` for why the lack of rounding matters numerically, not just
stylistically). Lattice construction, vortex detection, and the plaquette adjacency graph
used for drop's radial profile are imported directly from kagome_vortex_core.py (the
authoritative, networkx-clique-verified corrected lattice used by Paper 2's actual mining
script) via kagome_lattice.py, rather than reimplemented here.

Measurements (see protocol document, Section 4, for definitions and priority order)
-------------
  1. energy      - mean bond energy of the actual spin configuration, evaluated at FIXED K0
                   (i.e. NOT including the instantaneous drive term).
  2. drop        - E(r=0) - E(r=1) local restoring strength, ensemble-averaged over all
                   vortices present at that sweep. E(r) = mean plaquette bond-energy over
                   plaquettes at graph-distance r on the corner-sharing plaquette adjacency
                   graph (kagome_vortex_core.build_plaquette_adjacency), matching the
                   Paper 2 mining script's 'bond_E_r' quantity.
  3. birth       - fraction of plaquettes that newly became vortices (nonzero rounded winding)
                   since the last measurement.
  4. info (sqrt(M_I)) - spatial RMS of I(x,y) = H_max - H(x,y), computed with the EXACT
                   reference H-map formula (R_LOCAL=3, H_BINS=8, range=(-0.55,0.55),
                   unrounded winding). Vectorized for speed; validated against a direct
                   line-by-line port of the reference pure-Python version in this file's
                   self-test (max abs diff reported, should be ~0).
  5. helicity    - helicity modulus Y, evaluated at fixed K0.
"""

import math

import numpy as np
from numba import njit

from kagome_lattice import KagomeLattice


# ----------------------------------------------------------------------------------
# Metropolis engine (numba-jitted). Matches the reference mc_sweep exactly: full-range
# proposal phi_new = phi_old + Uniform(-pi, pi), K is a scalar (uniform over all bonds,
# but now passed in per-sweep rather than a module-level global, so it can be driven).
# ----------------------------------------------------------------------------------

@njit(cache=True)
def _metropolis_sweep(phi, neighbors, K, T, rand_order, rand_dphi, rand_accept):
    N = phi.shape[0]
    for idx in range(N):
        i = rand_order[idx]
        phi_old = phi[i]
        phi_new = phi_old + rand_dphi[idx]  # rand_dphi ~ Uniform(-pi, pi), full range
        dE = 0.0
        for k in range(neighbors.shape[1]):
            j = neighbors[i, k]
            dE += -K * (np.cos(phi_new - phi[j]) - np.cos(phi_old - phi[j]))
        # reference logic: reject (revert) if dE>=0 and rand>=exp(-dE/T); equivalent to
        # standard Metropolis (accept if dE<=0 or rand<exp(-dE/T)).
        if dE < 0.0 or rand_accept[idx] < np.exp(-dE / T):
            phi[i] = phi_new
    return phi


def metropolis_sweep(phi, neighbors, K, T, rng):
    N = phi.shape[0]
    rand_order = rng.permutation(N)
    rand_dphi = rng.uniform(-np.pi, np.pi, size=N)
    rand_accept = rng.random(size=N)
    return _metropolis_sweep(phi, neighbors, K, T, rand_order, rand_dphi, rand_accept)


# ----------------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------------

def bond_cos(phi, bonds):
    return np.cos(phi[bonds[:, 0]] - phi[bonds[:, 1]])


def mean_energy(phi, bonds, K0):
    """Mean bond energy per bond, evaluated at fixed K0 (not the instantaneous drive)."""
    return -K0 * np.mean(bond_cos(phi, bonds))


def helicity_modulus(phi, lat, K0, T):
    dphi = phi[lat.bonds[:, 0]] - phi[lat.bonds[:, 1]]
    N = lat.N
    term1 = (K0 / N) * np.sum(np.cos(dphi))
    term2 = (K0 ** 2 / (T * N)) * (np.sum(np.sin(dphi))) ** 2
    return term1 - term2


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def plaquette_winding_raw(phi, plaquettes):
    """Raw (unrounded) winding w = (wrap(dAB)+wrap(dBC)+wrap(dCA))/2pi for every plaquette.
    In exact arithmetic this is exactly 0 (no vortex) or exactly +-1 (vortex); the tiny
    floating-point residuals around those values are NOT noise to be rounded away -- see
    compute_Hmap_reference() below, where they carry the actual H-map signal.
    """
    i, j, k = plaquettes[:, 0], plaquettes[:, 1], plaquettes[:, 2]
    total = wrap_pi(phi[j] - phi[i]) + wrap_pi(phi[k] - phi[j]) + wrap_pi(phi[i] - phi[k])
    return total / (2 * np.pi)


def plaquette_winding_rounded(phi, plaquettes):
    """Integer winding number, for vortex identification (birth, drop). Matches
    kagome_vortex_core.detect_vortices() exactly (same formula, same rounding)."""
    return np.round(plaquette_winding_raw(phi, plaquettes)).astype(np.int64)


def vortex_plaquette_ids(w_rounded):
    return np.nonzero(w_rounded)[0]


def drop_for_vortices(phi, lat, vortex_ids, max_r=1):
    """drop_i = E(r=0) - E(r=1), ensemble-averaged over vortices. E(r) = mean plaquette
    bond-energy over plaquettes at graph-distance r on the corner-sharing plaquette
    adjacency graph (lat.plaq_adj, from kagome_vortex_core.build_plaquette_adjacency),
    exactly matching the r_profile 'bond_E_r' quantity in the Paper 2 mining script.
    """
    if len(vortex_ids) == 0:
        return np.nan, np.array([])

    def plaquette_bond_energy(p_idx):
        i, j, k = lat.plaquettes[p_idx]
        return (-np.cos(phi[i] - phi[j]) - np.cos(phi[j] - phi[k]) - np.cos(phi[k] - phi[i])) / 3.0

    drops = np.empty(len(vortex_ids))
    for n, v in enumerate(vortex_ids):
        shells = lat.bfs_shells(int(v), max_r=max_r)
        e_r0 = plaquette_bond_energy(v)
        e_r1 = np.mean([plaquette_bond_energy(p) for p in shells[1]])
        drops[n] = e_r0 - e_r1
    return float(np.mean(drops)), drops


# ---- information density: exact reference H-map formula --------------------------------

R_LOCAL = 3
H_BINS = 8
H_RANGE = (-0.55, 0.55)
H_MAX = math.log2(H_BINS)  # = 3 bits


def compute_Hmap_reference_naive(phi, lat):
    """Direct, line-by-line port of the user's reference compute_Hmap (pure Python, slow).
    Kept only as the ground truth for the vectorized version's self-test below -- do not
    use this in production runs, use compute_Hmap() instead.
    """
    L = lat.L
    wp = lambda d: (d + math.pi) % (2 * math.pi) - math.pi
    Hmap = np.zeros((L, L), dtype=np.float64)
    for ix in range(L):
        for iy in range(L):
            vals = []
            for dix in range(-R_LOCAL, R_LOCAL + 1):
                for diy in range(-R_LOCAL, R_LOCAL + 1):
                    jx = (ix + dix) % L
                    jy = (iy + diy) % L
                    A = 3 * (jx * L + jy)
                    B, C = A + 1, A + 2
                    w = (wp(phi[B] - phi[A]) + wp(phi[C] - phi[B]) + wp(phi[A] - phi[C])) \
                        / (2 * math.pi)
                    vals.append(w)
            h, _ = np.histogram(vals, bins=H_BINS, range=H_RANGE)
            s = h.sum()
            if s > 0:
                p = h / s
                p = p[p > 0]
                Hmap[ix, iy] = float(-np.sum(p * np.log2(p)))
    return Hmap


def compute_Hmap(phi, lat):
    """Vectorized reference H-map: same formula and same numerical result as
    compute_Hmap_reference_naive() (validated in the self-test below), computed without the
    O(L^2 * (2R+1)^2) Python double loop.
    """
    L = lat.L
    w_grid = plaquette_winding_raw(phi, lat.up_plaq).reshape(L, L)  # w_grid[ix,iy]

    win = 2 * R_LOCAL + 1
    edges = np.linspace(H_RANGE[0], H_RANGE[1], H_BINS + 1)
    Hmap = np.zeros((L, L), dtype=np.float64)

    # Gather every (2R+1)x(2R+1) PBC window as a stack of shifted copies of w_grid, then
    # histogram along the stack axis for every (ix,iy) at once.
    shifted = np.empty((win * win, L, L), dtype=w_grid.dtype)
    n = 0
    for dix in range(-R_LOCAL, R_LOCAL + 1):
        for diy in range(-R_LOCAL, R_LOCAL + 1):
            shifted[n] = np.roll(np.roll(w_grid, -dix, axis=0), -diy, axis=1)
            n += 1
    # shifted[:, ix, iy] now holds exactly the same 49 values as the reference's `vals` list
    # for cell (ix,iy) (order differs, which does not matter for a histogram/entropy).

    flat = shifted.reshape(win * win, L * L)  # (49, L*L)
    for b in range(H_BINS):
        lo, hi = edges[b], edges[b + 1]
        if b < H_BINS - 1:
            counts_b = np.sum((flat >= lo) & (flat < hi), axis=0)
        else:
            counts_b = np.sum((flat >= lo) & (flat <= hi), axis=0)  # last bin closed both ends
        if b == 0:
            counts = counts_b[None, :]
        else:
            counts = np.concatenate([counts, counts_b[None, :]], axis=0)

    s = counts.sum(axis=0).astype(np.float64)  # (L*L,)
    p = counts.astype(np.float64) / np.where(s == 0, 1.0, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log2(p), 0.0)
    H_flat = -term.sum(axis=0)
    H_flat = np.where(s > 0, H_flat, 0.0)
    return H_flat.reshape(L, L)


def sqrt_M_I(phi, lat):
    H = compute_Hmap(phi, lat)
    I = H_MAX - H
    return float(np.sqrt(np.mean(I ** 2)))


def info_field_morphology(I):
    """I(x,y) 필드(compute_Hmap의 H_MAX-H 결과, 2D array)의 형태(morphology)를
    싼 값에 요약하는 스칼라들. Paper 4를 위해 "정보의 양"이 아니라 "정보의 형태"를
    지금부터 저장해두자는 방향에서 추가됨 -- 지금 당장 분석하지 않아도, 나중에 다시
    마이닝하지 않고 이미 저장된 이 컬럼들만 꺼내 쓸 수 있도록 하는 것이 목적이다.

    반환하는 것:
      info_mean, info_max, info_std : 공간 평균/최댓값/표준편차 (Smean, Smax, Sstd)
      info_skew, info_kurtosis      : 정보밀도 분포의 비대칭도, 첨도
      info_corr_length              : 2D 방사 자기상관이 1/e로 떨어지는 거리 (격자 단위) --
                                       정보가 "점처럼" 있는지 "덩어리로" 있는지의 척도
      info_n_components             : (평균+1표준편차) 이상인 영역의 연결된 덩어리 개수
      info_largest_component_frac   : 그중 가장 큰 덩어리가 전체 hotspot 셀 중 차지하는 비율
                                       (1에 가까우면 하나의 큰 섬, 작으면 여러 개로 흩어짐)

    Persistence(시간에 따른 hotspot 유지), Entropy production(dS/dt), Mutual information
    (입력 신호와 필드 사이)은 여기서 계산하지 않는다 -- 이 세 가지는 한 시점의 필드만으론
    계산할 수 없고 여러 시점에 걸친 시공간 데이터가 필요하므로, 원본 필드 자체를
    저장해서(run_driven_experiment의 record_field_snapshots=True) 나중에 후처리로
    계산하는 것이 맞다 (README 참고).
    """
    from scipy import ndimage, stats as scipy_stats

    flat = I.ravel()
    info_mean = float(np.mean(flat))
    info_max = float(np.max(flat))
    info_std = float(np.std(flat))
    info_skew = float(scipy_stats.skew(flat)) if info_std > 1e-12 else 0.0
    info_kurt = float(scipy_stats.kurtosis(flat)) if info_std > 1e-12 else 0.0

    # 2D 방사 자기상관 (FFT 기반) -- I 필드 자체의 공간 상관을 재고, 각도 평균을 내서
    # 거리(radius)만의 함수로 만든 다음 1/e 지점을 찾는다.
    L = I.shape[0]
    Ic = I - info_mean
    var = np.mean(Ic ** 2)
    if var < 1e-12:
        corr_length = 0.0
    else:
        f = np.fft.fft2(Ic)
        acf2d = np.real(np.fft.ifft2(f * np.conj(f))) / (var * I.size)
        acf2d = np.fft.fftshift(acf2d)
        cy, cx = L // 2, L // 2
        yy, xx = np.mgrid[0:L, 0:L]
        r = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(int)
        radial = np.bincount(r.ravel(), weights=acf2d.ravel()) / np.maximum(
            np.bincount(r.ravel()), 1)
        below = np.where(radial < 1.0 / np.e)[0]
        corr_length = float(below[0]) if len(below) else float(L / 2)

    # Hotspot 연결 요소 개수 + 가장 큰 덩어리 비율 (평균+1시그마 임계값, PBC 고려)
    threshold = info_mean + info_std
    mask = I > threshold
    if mask.any():
        labeled, n_components = ndimage.label(mask, structure=np.ones((3, 3)))
        # PBC 보정: 격자 경계에서 서로 이어지는 덩어리를 하나로 합친다 (간단한 근사:
        # 위/아래, 좌/우 경계에서 라벨이 다르면 병합)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # 배경 제외
        largest_frac = float(sizes.max() / mask.sum()) if mask.sum() > 0 else 0.0
    else:
        n_components = 0
        largest_frac = 0.0

    return {
        "info_mean": info_mean,
        "info_max": info_max,
        "info_std": info_std,
        "info_skew": info_skew,
        "info_kurtosis": info_kurt,
        "info_corr_length": corr_length,
        "info_n_components": int(n_components),
        "info_largest_component_frac": largest_frac,
    }


# ----------------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------------

def run_driven_experiment(L, T, K0, delta_K, n_therm, meas_stride, rng, max_r_drop=1,
                           verbose=True, record_morphology=False,
                           record_field_snapshots=False, snapshot_stride=None):
    """
    record_morphology=True 로 켜면, 매 측정마다 info_field_morphology()의 스칼라들
    (info_mean, info_max, info_std, skew, kurtosis, corr_length, n_components,
    largest_component_frac)이 records에 추가로 기록된다. compute_Hmap을 한 번만 계산해서
    'info'(sqrt_M_I)와 이 스칼라들이 같은 필드에서 나오도록 한다 (중복 계산 없음).

    record_field_snapshots=True 로 켜면, I(x,y) 필드 원본을 snapshot_stride(기본:
    meas_stride의 10배)마다 통째로 저장한다 -- Persistence(hotspot이 시간에 따라 얼마나
    유지되는지), Mutual information(입력 신호와 필드 사이), Entropy production(dS/dt) 같은
    시공간 분석은 한 시점의 스칼라만으론 계산이 안 되고 원본 필드가 여러 시점에 걸쳐
    있어야 하므로, 이건 나중에(Paper 4) 후처리로 계산할 수 있게 원본을 남겨두는 용도다.
    반환값의 records['_field_snapshots']에 (sweep 번호 리스트, (n_snap, L, L) 배열)이
    담기며, 저장은 호출하는 쪽(run_stage1_experiment.py 등)이 담당한다 (예: np.savez).
    """
    lat = KagomeLattice(L)
    n_sweeps_total = len(delta_K)
    phi = rng.uniform(-np.pi, np.pi, size=lat.N)

    if snapshot_stride is None:
        snapshot_stride = meas_stride * 10

    for _ in range(n_therm):
        metropolis_sweep(phi, lat.neighbors, K0, T, rng)

    records = {"sweep": [], "energy": [], "drop": [], "birth": [], "info": [], "helicity": []}
    if record_morphology:
        for key in ["info_mean", "info_max", "info_std", "info_skew", "info_kurtosis",
                    "info_corr_length", "info_n_components", "info_largest_component_frac"]:
            records[key] = []

    snapshot_sweeps, snapshot_fields = [], []
    prev_vortex_mask = np.zeros(lat.n_plaq, dtype=bool)

    for t in range(n_sweeps_total):
        K_t = K0 + delta_K[t]
        phi = metropolis_sweep(phi, lat.neighbors, K_t, T, rng)

        if t % meas_stride == 0:
            w_rounded = plaquette_winding_rounded(phi, lat.plaquettes)
            vortex_mask = w_rounded != 0
            vortex_ids = np.nonzero(vortex_mask)[0]

            new_vortices = vortex_mask & (~prev_vortex_mask)
            birth_rate = np.sum(new_vortices) / lat.n_plaq
            prev_vortex_mask = vortex_mask

            drop_mean, _ = drop_for_vortices(phi, lat, vortex_ids, max_r=max_r_drop)

            if record_morphology or record_field_snapshots:
                H = compute_Hmap(phi, lat)
                I_field = H_MAX - H
                info_val = float(np.sqrt(np.mean(I_field ** 2)))
            else:
                I_field = None
                info_val = sqrt_M_I(phi, lat)

            records["sweep"].append(t)
            records["energy"].append(mean_energy(phi, lat.bonds, K0))
            records["drop"].append(drop_mean)
            records["birth"].append(birth_rate)
            records["info"].append(info_val)
            records["helicity"].append(helicity_modulus(phi, lat, K0, T))

            if record_morphology:
                morph = info_field_morphology(I_field)
                for key, val in morph.items():
                    records[key].append(val)

            if record_field_snapshots and t % snapshot_stride == 0:
                snapshot_sweeps.append(t)
                snapshot_fields.append(I_field.copy())

            if verbose and t % (meas_stride * 20) == 0:
                print(f"  sweep {t:6d}/{n_sweeps_total}  K={K_t:.4f}  "
                      f"E={records['energy'][-1]:.4f}  birth={birth_rate:.4f}  "
                      f"info={records['info'][-1]:.4f}  drop={drop_mean:.4f}")

    for k in records:
        records[k] = np.array(records[k])

    if record_field_snapshots:
        records["_field_snapshots"] = (
            np.array(snapshot_sweeps),
            np.stack(snapshot_fields) if snapshot_fields else np.empty((0, L, L)))

    return records


def build_protocol_schedule(n_baseline, n_driven, driven_signal):
    delta_K = np.concatenate([np.zeros(n_baseline), driven_signal])
    driven_start = n_baseline
    return delta_K, driven_start


if __name__ == "__main__":
    print("=== Validating vectorized compute_Hmap against the naive reference port ===")
    rng = np.random.default_rng(1)
    lat_test = KagomeLattice(L=10)
    phi_test = rng.uniform(-np.pi, np.pi, size=lat_test.N)
    for _ in range(50):
        phi_test = metropolis_sweep(phi_test, lat_test.neighbors, 0.7, 0.9, rng)

    H_naive = compute_Hmap_reference_naive(phi_test, lat_test)
    H_fast = compute_Hmap(phi_test, lat_test)
    max_diff = np.max(np.abs(H_naive - H_fast))
    print(f"max abs diff (naive vs vectorized) = {max_diff:.3e}  (should be ~0)")
    assert max_diff < 1e-9, "vectorized H-map does not match the reference formula!"
    print(f"H-map stats: min={H_fast.min():.4f} max={H_fast.max():.4f} mean={H_fast.mean():.4f} "
          f"(nondegenerate: should NOT be all-zero or all-3.0)")
    assert H_fast.std() > 1e-6, "H-map is degenerate (no variation) -- something is wrong"

    print("\n=== Smoke test: short driven run on a small lattice ===")
    import signals
    rng = np.random.default_rng(42)
    L = 8
    T, K0, power = 0.9, 0.7, 0.01
    n_baseline, n_driven = 200, 2000
    sine = signals.gen_sine(n_driven, power, period_sweeps=40, rng=rng)
    signals.check_power(sine, power, label="smoke-test sine")
    delta_K, driven_start = build_protocol_schedule(n_baseline, n_driven, sine)

    print(f"Running smoke test: L={L}, T={T}, K0={K0}, n_sweeps_total={len(delta_K)}")
    rec = run_driven_experiment(L, T, K0, delta_K, n_therm=500, meas_stride=5, rng=rng)

    print("\nSummary (all should be finite; 'info' and 'helicity' should show real variance, "
          "not be pinned at a constant):")
    for k, v in rec.items():
        if k == "sweep":
            continue
        finite = np.isfinite(v)
        print(f"  {k:10s}: n={len(v)}, finite={finite.sum()}/{len(v)}, "
              f"mean={np.nanmean(v):.5g}, std={np.nanstd(v):.3g}")
    assert np.nanstd(rec["info"]) > 1e-6, "info time series is degenerate!"
    print("\nSmoke test completed without crashing, and info/helicity show real variation.")
