"""
kagome_fss_v3.py — FSS 스캔 v3 (벡터화 + 배치 마이닝)
============================================================
v2 대비 변경점:
  1. compute_hmap, vortex_obs를 넘파이로 벡터화 (검증 완료:
     vectorize_check.py에서 L=8,12,16,24 / 여러 T,seed로
     기존 v2 결과와 수치 완전 일치 확인됨 — max diff = 0.0)
  2. --seed_start/--n_seeds 또는 --seeds로 여러 시드를 한 번에
     순차 마이닝 가능 (예: seed 1~30을 한 번의 실행으로 처리)
  3. --L은 기존처럼 임의 격자 크기 지정 가능 (32,48,64,96,128,256...)
  4. 시드별로 독립된 출력 CSV (체크포인트도 시드별로 분리)
     → 중간에 끊겨도 이미 끝난 시드/T는 건너뜀

실행 예:
  # 단일 시드 (기존과 동일)
  python kagome_fss_v3.py --L 128 --seed 111 --mode broad

  # 시드 1~30 배치 마이닝 (128 격자, 30개 시드 한 번에)
  python kagome_fss_v3.py --L 128 --seed_start 1 --n_seeds 30 --mode broad

  # 특정 시드 목록만 지정
  python kagome_fss_v3.py --L 128 --seeds 101,102,105,200 --mode fine

  # 256 격자도 동일하게 --L만 바꿔서 사용
  python kagome_fss_v3.py --L 256 --seed_start 1 --n_seeds 30 --mode broad
============================================================
"""
import os, sys, math, time, argparse, csv

# Windows CMD에서 한글 print 시 UnicodeEncodeError(cp1252) 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import laplace, gaussian_filter
from numpy.lib.stride_tricks import sliding_window_view

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("경고: numba 미설치 — 'pip install numba --break-system-packages'로 설치하면"
          " mc_sweep이 10~60배 빨라집니다. 지금은 느린 순수 파이썬 버전으로 동작합니다.")

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

CFG = dict(N_THERM=800, N_SNAP=120, SNAP_EVERY=10)

MODES = {
    'broad': (0.85, 1.61, 0.02),    # 39점
    'fine':  (0.97, 1.121, 0.005),  # 31점 — T≈1.05 정밀
}

wp = lambda d: (d + math.pi) % (2*math.pi) - math.pi


# ============================================================
# 격자 구조 (L 임의 지정 가능 — 기존과 동일)
# ============================================================
def build_bonds(L):
    all_bonds = []; x_bonds = []
    for ix in range(L):
        for iy in range(L):
            base = 3*(ix*L+iy); A, B, C = base, base+1, base+2
            all_bonds.append((A, B)); x_bonds.append((A, B))
            all_bonds.append((B, C)); all_bonds.append((C, A))
            nx = 3*(((ix+1) % L)*L+iy)
            all_bonds.append((A, nx+1)); x_bonds.append((A, nx+1))
            ny = 3*(ix*L+(iy+1) % L)
            all_bonds.append((B, ny+2))
            nd = 3*(((ix+1) % L)*L+(iy+1) % L)
            all_bonds.append((C, nd))
    return (np.array(all_bonds),
            np.array(x_bonds) if x_bonds else np.empty((0, 2), dtype=int))


def precompute_neighbors(N, bonds):
    """기존 호환용 (bond 인덱스 리스트). 더 이상 mc_sweep에는 안 쓰이지만
    다른 코드와의 호환성을 위해 유지."""
    nb = [[] for _ in range(N)]
    for b, (i, j) in enumerate(bonds):
        nb[i].append(b); nb[j].append(b)
    return nb


def build_fixed_neighbors(L):
    """각 사이트의 '이웃 사이트' 인덱스를 (N,4) 고정폭 배열로 미리 펼침
    (카고메 격자는 모든 사이트가 정확히 4개 결합 → Numba 벡터 연산에 최적).
    수치적으로 기존 mc_sweep(bond 인덱스 경유 방식)과 완전히 동일한 결과를
    내는 것이 mc_sweep_numba.py에서 검증됨(여러 L, 여러 시드, 460+ sweep
    누적 후에도 helicity 값 소수점까지 완전 일치, max diff = 0.0)."""
    bonds, _ = build_bonds(L)
    N = 3 * L * L
    neighbor_sites = [[] for _ in range(N)]
    for (i, j) in bonds:
        neighbor_sites[i].append(j)
        neighbor_sites[j].append(i)
    return np.array(neighbor_sites, dtype=np.int64)  # (N, 4)


# ============================================================
# Monte Carlo sweep — Numba 가속
# (기존 kagome_bkt_v2.mc_sweep과 수치적으로 완전히 동등함 검증됨.
#  L=128: 약 11~13배, L=256: 약 60배 이상 빠름)
# ============================================================
if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _mc_sweep_numba_core(phi, K, nb_fixed, T, rand_sites, rand_deltas, rand_accepts):
        N = phi.shape[0]
        for k in range(N):
            site = rand_sites[k]
            phi_site = phi[site]
            E_old = 0.0
            E_new = 0.0
            phi_trial = phi_site + rand_deltas[k]
            for d in range(4):
                j = nb_fixed[site, d]
                E_old -= K * math.cos(phi_site - phi[j])
                E_new -= K * math.cos(phi_trial - phi[j])
            dE = E_new - E_old
            if dE < 0.0 or rand_accepts[k] < math.exp(-dE / T):
                phi[site] = phi_trial
        return phi

    def mc_sweep(phi, K, nb_fixed, T, rng):
        """기존 mc_sweep과 동일 시그니처(nb_fixed는 build_fixed_neighbors 결과).
        RNG 소비: permutation → uniform(delta, N개) → random(accept, N개) 순서로
        항상 둘 다 뽑음 (기존 코드의 short-circuit과 다르지만 물리적으로 동등,
        검증 완료: mc_sweep_numba.py 참고)."""
        N = len(phi)
        sites = rng.permutation(N)
        deltas = rng.uniform(-np.pi, np.pi, N)
        accepts = rng.random(N)
        return _mc_sweep_numba_core(phi, K, nb_fixed, T, sites, deltas, accepts)
else:
    def mc_sweep(phi, K, nb_fixed, T, rng):
        """numba 미설치 시 순수 파이썬 fallback (느림)."""
        N = len(phi)
        sites = rng.permutation(N)
        deltas = rng.uniform(-np.pi, np.pi, N)
        accepts = rng.random(N)
        for k in range(N):
            site = sites[k]
            phi_site = phi[site]
            E_old = 0.0
            for d in range(4):
                j = nb_fixed[site, d]
                E_old -= K * math.cos(phi_site - phi[j])
            phi_trial = phi_site + deltas[k]
            E_new = 0.0
            for d in range(4):
                j = nb_fixed[site, d]
                E_new -= K * math.cos(phi_trial - phi[j])
            dE = E_new - E_old
            if dE < 0 or accepts[k] < math.exp(-dE / T):
                phi[site] = phi_trial
        return phi


def measure_helicity(phi, K, xb, T):
    if len(xb) == 0:
        return 0.0
    L2 = len(phi)//3
    bi, bj = xb[:, 0], xb[:, 1]
    dphi = phi[bi]-phi[bj]
    return (K/L2)*np.sum(np.cos(dphi)) - (K**2/(T*L2))*np.sum(np.sin(dphi))**2


# ============================================================
# 벡터화된 윈딩맵 / hmap / vortex_obs
# (vectorize_check.py로 기존 v2와 수치 완전 일치 검증됨)
# ============================================================
def kagome_winding_map(phi, L):
    """전체 격자 winding number를 한 번에 계산 → (L,L) 배열"""
    phi3 = phi.reshape(L, L, 3)
    A, B, C = phi3[..., 0], phi3[..., 1], phi3[..., 2]
    dAB = (B - A + np.pi) % (2*np.pi) - np.pi
    dBC = (C - B + np.pi) % (2*np.pi) - np.pi
    dCA = (A - C + np.pi) % (2*np.pi) - np.pi
    return (dAB + dBC + dCA) / (2*np.pi)


def compute_hmap(phi, L, bins=8, r=3):
    """국소 엔트로피 맵 (벡터화). np.histogram(range=(-0.55,0.55))과
    동일하게 범위 밖 값은 버림(clip 금지) — 보텍스(±1.0)가 있는
    윈도우는 항상 그 값이 통계에서 제외됨, 기존 v2와 동일 동작."""
    W = kagome_winding_map(phi, L)
    win = 2*r + 1
    Wp = np.pad(W, r, mode='wrap')
    windows = sliding_window_view(Wp, (win, win))   # (L, L, win, win)
    flat = windows.reshape(L, L, -1)

    edges = np.linspace(-0.55, 0.55, bins+1)
    in_range = (flat >= edges[0]) & (flat <= edges[-1])
    idx = np.digitize(flat, edges[1:-1])             # 0..bins-1
    idx = np.where(in_range, idx, -1)                # 범위 밖 → 어떤 bin에도 미포함

    counts = np.zeros((L, L, bins), dtype=np.int64)
    for b in range(bins):
        counts[..., b] = np.sum(idx == b, axis=-1)

    s = counts.sum(axis=-1, keepdims=True).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(s > 0, counts/np.where(s == 0, 1, s), 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    H = -np.sum(p*logp, axis=-1).astype(np.float32)
    return H


def vortex_obs(phi, L):
    """Nv, near_frac, Smax — PBC 클러스터링 정확히 반영 (검증 완료)"""
    W = kagome_winding_map(phi, L)
    mask = np.abs(W) > 0.45
    n_vortex = int(mask.sum())
    Nv = n_vortex/(L*L)

    shifted_or = (np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0) |
                  np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1))
    near = np.sum(mask & shifted_or)
    nf = near/max(n_vortex, 1)

    # Smax: PBC BFS (scipy.ndimage.label은 PBC 미지원이라 직접 구현 유지)
    vm_set = set(zip(*np.nonzero(mask)))
    visited = set(); smax = 0
    for pos in vm_set:
        if pos in visited:
            continue
        q = [pos]; visited.add(pos); sz = 0
        while q:
            cx, cy = q.pop(); sz += 1
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                npos = ((cx+dx) % L, (cy+dy) % L)
                if npos in vm_set and npos not in visited:
                    visited.add(npos); q.append(npos)
        smax = max(smax, sz)

    return Nv, nf, smax


def hmap_obs(H, L):
    I = H.max()-H
    sqrtMI = float(np.sqrt(np.mean(I**2)))
    RI = float(np.abs(laplace(gaussian_filter(I, sigma=1.0))).mean())
    Hc = H.astype(np.float64)-H.mean()
    F = np.fft.fft2(Hc)
    C2d = np.real(np.fft.ifft2(F*np.conj(F)))/(L*L)
    ix_g = np.arange(L); iy_g = np.arange(L)
    dx, dy = np.meshgrid(ix_g, iy_g, indexing='ij')
    dx = np.where(dx > L//2, dx-L, dx); dy = np.where(dy > L//2, dy-L, dy)
    r_f = np.sqrt(dx**2+dy**2).flatten(); c_f = C2d.flatten()
    r_bins = np.arange(0.5, L//2+0.5, 1.0)
    C_r = np.array([c_f[(r_f >= rb-0.5) & (r_f < rb+0.5)].mean()
                     if ((r_f >= rb-0.5) & (r_f < rb+0.5)).sum() > 0
                     else np.nan for rb in r_bins])
    pos = (~np.isnan(C_r)) & (C_r > 0) & (r_bins > 0)
    if pos.sum() >= 2:
        den = trapz(C_r[pos], r_bins[pos])+1e-10
        xm1 = float(trapz(r_bins[pos]*C_r[pos], r_bins[pos])/den)
    else:
        xm1 = float('nan')
    return sqrtMI, RI, xm1


def progress_bar(cur, total, width=25, extra=''):
    pct = cur/total; filled = int(width*pct)
    bar = '█'*filled+'░'*(width-filled)
    print(f"\r  [{bar}] {cur:3d}/{total} {pct*100:4.1f}%  {extra}",
          end='', flush=True)


# ============================================================
# 시드 1개 처리 (기존 main 로직)
# ============================================================
def run_one_seed(L, SEED, T_list, N_THERM, N_SNAP, SNAP_EVERY, out, mode_label):
    N = 3*L*L; K = 1.0

    fields = ['T', 'seed', 'L',
              'Y', 'YoverT',
              'sqrtMI', 'absRI', 'xm1',
              'birth', 'Smax', 'Nv',
              'eta']

    bonds, x_bonds = build_bonds(L)
    nb = build_fixed_neighbors(L)

    print(f"{'='*60}")
    print(f"  FSS v3 (벡터화)  L={L}  seed={SEED}  mode={mode_label}")
    print(f"  T={T_list[0]:.3f}~{T_list[-1]:.3f}  ({len(T_list)}점)")
    print(f"  N={N}  N_SNAP={N_SNAP}")
    print(f"  출력: {out}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(SEED)

    # 체크포인트: 이미 완료된 T 확인 + phi/rng 상태 복원
    state_path = out + '.state.npz'
    import pandas as _pd
    done_T = set()
    if os.path.exists(out):
        try:
            df_done = _pd.read_csv(out)
            done_T = set(df_done['T'].round(4).tolist())
            print(f"  ★ 체크포인트: {len(done_T)}개 T 완료 → 이어서 시작")
        except Exception:
            done_T = set()
    T_remain = [T for T in T_list if round(T, 4) not in done_T]
    if not T_remain:
        print("  이 시드는 모든 T 완료됨 → 건너뜀\n")
        return
    T_list_run = T_remain

    resumed = False
    if done_T and os.path.exists(state_path):
        try:
            saved = np.load(state_path, allow_pickle=True)
            phi = saved['phi']
            rng_state = saved['rng_state'].item()
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state
            resumed = True
            print(f"  ★ 이전 phi/rng 상태 복원 → 열화 스킵, 바로 이어서 진행")
        except Exception as e:
            print(f"  ! 상태 파일 복원 실패({e}) → 처음부터 열화")

    if not resumed:
        phi = rng.uniform(-math.pi, math.pi, N)
        print(f"  열화 중 ({N_THERM} sweeps)...", flush=True)
        t0 = time.time()
        for _ in range(N_THERM):
            phi = mc_sweep(phi, K, nb, T_list_run[0], rng)
        print(f"  완료 ({time.time()-t0:.1f}s)\n")

    file_mode = 'a' if done_T else 'w'
    with open(out, file_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done_T:
            writer.writeheader()

        for ti, T in enumerate(T_list_run):
            for _ in range(100):
                phi = mc_sweep(phi, K, nb, T, rng)

            Y_s = []; H_snaps = []
            Nv_s = []; birth_s = []; smax_s = []
            t_start = time.time()

            for si in range(N_SNAP):
                for _ in range(SNAP_EVERY):
                    phi = mc_sweep(phi, K, nb, T, rng)
                Y_s.append(measure_helicity(phi, K, x_bonds, T))
                H_snaps.append(compute_hmap(phi, L))
                nv, br, sm = vortex_obs(phi, L)
                Nv_s.append(nv); birth_s.append(br); smax_s.append(sm)

                elapsed = time.time()-t_start
                eta_t = elapsed/(si+1)*(N_SNAP-si-1)
                progress_bar(si+1, N_SNAP,
                    extra=f'T={T:.3f} ETA:{eta_t:.0f}s '
                          f'Y/T={np.mean(Y_s)/T:.4f}')

            print()
            Y_m = float(np.mean(Y_s))
            H_mean = np.mean(H_snaps, axis=0)
            sqrtMI, RI, xm1 = hmap_obs(H_mean, L)
            br_m = float(np.mean(birth_s))
            sm_m = float(np.mean(smax_s))
            nv_m = float(np.mean(Nv_s))
            eta = br_m/(sqrtMI+1e-10)

            writer.writerow(dict(
                T=T, seed=SEED, L=L,
                Y=round(Y_m, 6), YoverT=round(Y_m/T, 6),
                sqrtMI=round(sqrtMI, 6), absRI=round(RI, 7),
                xm1=(round(xm1, 5) if not math.isnan(xm1) else ''),
                birth=round(br_m, 6), Smax=round(sm_m, 4),
                Nv=round(nv_m, 6),
                eta=round(eta, 4),
            ))
            f.flush()

            # 다음 재개를 위해 phi/rng 상태 저장
            np.savez(state_path, phi=phi, rng_state=rng.bit_generator.state)

            print(f"  → T={T:.3f}  Y/T={Y_m/T:.4f}  "
                  f"sqMI={sqrtMI:.4f}  birth={br_m:.4f}  "
                  f"eta={eta:.3f}\n", flush=True)

    print(f"  완료: {out}\n")


# ============================================================
# 메인: 다중 시드 배치 마이닝
# ============================================================
def parse_seed_list(args):
    """--seeds 가 있으면 그걸 우선, 없으면 --seed_start/--n_seeds로 생성,
    둘 다 없으면 --seed 단일 사용."""
    if args.seeds:
        return [int(s) for s in args.seeds.split(',') if s.strip() != '']
    if args.n_seeds and args.n_seeds > 1:
        start = args.seed_start if args.seed_start is not None else args.seed
        return list(range(start, start + args.n_seeds))
    return [args.seed]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, required=True,
                   help='격자 크기 (32,48,64,96,128,256... 임의 지정 가능)')
    p.add_argument('--seed', type=int, default=111,
                   help='단일 시드 (배치 옵션 미지정 시 사용)')
    p.add_argument('--seed_start', type=int, default=None,
                   help='배치 시작 시드 (--n_seeds와 함께 사용)')
    p.add_argument('--n_seeds', type=int, default=None,
                   help='배치로 돌릴 시드 개수 (예: 30 → seed_start부터 30개)')
    p.add_argument('--seeds', type=str, default=None,
                   help='쉼표로 구분된 시드 목록 (예: "101,102,105,200"), '
                        '지정 시 seed_start/n_seeds보다 우선')
    p.add_argument('--mode', choices=['broad', 'fine'], default='broad')
    p.add_argument('--T_min', type=float, default=None)
    p.add_argument('--T_max', type=float, default=None)
    p.add_argument('--dT', type=float, default=None)
    p.add_argument('--n_snap', type=int, default=CFG['N_SNAP'])
    p.add_argument('--n_therm', type=int, default=CFG['N_THERM'])
    p.add_argument('--out_dir', type=str, default='.',
                   help='시드별 출력 CSV를 저장할 디렉토리')
    p.add_argument('--out_prefix', type=str, default=None,
                   help='출력 파일명 접두사 (기본: fss_{mode}_L{L})')
    args = p.parse_args()

    L = args.L
    N_THERM = args.n_therm; N_SNAP = args.n_snap
    SNAP_EVERY = CFG['SNAP_EVERY']

    if args.T_min is not None:
        T_min, T_max, dT = args.T_min, args.T_max, args.dT or 0.02
    else:
        T_min, T_max, dT = MODES[args.mode]
    dec = 4 if dT < 0.01 else 3
    T_list = np.round(np.arange(T_min, T_max+1e-9, dT), dec)

    seeds = parse_seed_list(args)
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.out_prefix or f'fss_{args.mode}_L{L}'

    print(f"\n{'#'*60}")
    print(f"  배치 마이닝: L={L}  시드 {len(seeds)}개  "
          f"({seeds[0]}~{seeds[-1]})" if len(seeds) > 1 else
          f"  단일 시드: L={L}  seed={seeds[0]}")
    print(f"  T범위: {T_min}~{T_max} (dT={dT}, {len(T_list)}점)")
    print(f"{'#'*60}\n")

    t_batch_start = time.time()
    for n_done, SEED in enumerate(seeds, 1):
        out = os.path.join(args.out_dir, f'{prefix}_s{SEED}.csv')
        print(f"\n>>> [{n_done}/{len(seeds)}] seed={SEED} 시작 "
              f"(배치 경과 {time.time()-t_batch_start:.0f}s) <<<")
        run_one_seed(L, SEED, T_list, N_THERM, N_SNAP, SNAP_EVERY,
                     out, args.mode)

    print(f"\n{'#'*60}")
    print(f"  배치 마이닝 전체 완료: {len(seeds)}개 시드, "
          f"총 {time.time()-t_batch_start:.0f}s")
    print(f"  출력 디렉토리: {os.path.abspath(args.out_dir)}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
