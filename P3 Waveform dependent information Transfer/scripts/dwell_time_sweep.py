"""
Telegraph 신호의 dwell_sweeps(부호 유지 시간)를 스윕하는 실험.

Random telegraph 신호(+-A 두 값만 가짐)는 dwell_sweeps를 얼마로 바꾸든 입력 Shannon
entropy가 거의 항상 1비트 근처로 고정된다 (히스토그램에 값이 두 개뿐이므로). 즉
dwell_sweeps만 바꾸면 "입력 정보량(엔트로피)은 거의 그대로 두고, 입력의 시간적 상관구조
(correlation time)만 바꾸는" 실험이 자동으로 된다 -- 별도의 새 신호를 설계할 필요 없이,
이미 있는 telegraph 생성기의 파라미터 하나만 스윕하면 되는 것.

측정하는 것:
  1. 입력 신호 자체의 실측 자기상관시간 tau_signal (dwell_sweeps로부터 기대되는 값과
     대략 일치하는지 확인 -- 사인/백색잡음처럼 이론적으로 뻔한 게 아니라 실제 신호를
     만들어서 직접 재는 것)
  2. 입력 Shannon entropy (dwell_sweeps에 따라 거의 안 변한다는 것을 직접 확인)
  3. net TE(input -> Birth) (이게 이번 실험의 핵심 질문에 대한 답: entropy가 고정된
     상태에서 상관시간만 바뀌어도 TE가 달라지는가?)

TE가 tau_signal에 따라 유의미하게 변한다면, "격자는 정보량이 아니라 정보의 시간척도
(correlation time)에 반응한다"는 훨씬 강한 물리적 주장이 가능해진다.

사용법:
    python dwell_time_sweep.py --dwells 1,2,5,10,20,40 --seeds 0,1,2,3,4 \\
        --L 48 --T 0.9 --K0 0.7 --power 0.01 --n-baseline 5000 --n-driven 30000 \\
        --n-therm 45 --meas-stride 5 --burn-in-frac 0.0006 --outdir dwell_sweep_results
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from run_stage1_experiment import run_one_case, summarize_case
import signals


def measure_signal_tau(signal_1d, max_lag=200):
    """입력 신호 자체의 자기상관시간을 직접 측정 (Sokal 자동 윈도잉).
    pilot_calibration.py와 동일한 방법이지만, 여기서는 시뮬레이션 결과가 아니라
    입력 신호 자체에 대해 잰다는 점이 다르다.

    주의: 이 공식은 완벽하게 교대(anti-correlated)하는 신호(예: dwell_sweeps=1일 때의
    +A,-A,+A,-A,... 결정론적 사각파)에서는 깨진다 -- C(lag=1)이 정확히 -1이 되어
    tau = 0.5 + (-1) = -0.5 처럼 음수가 나온다. 이건 계산 실수가 아니라 Sokal 공식
    자체가 "느슨하게 상관된" 과정을 가정하고 있어서 극단적 반상관에서는 물리적으로
    의미 없는 값을 내는 것이다. 이런 경우를 위해 measure_dominant_period_fft()를
    같이 쓴다 -- FFT 기반 방법은 반상관 신호에서도 깨지지 않는다.
    """
    x = np.asarray(signal_1d, dtype=float)
    x = x - x.mean()
    var = np.mean(x ** 2)
    if var < 1e-15:
        return 0.5
    n = len(x)
    safe_max_lag = min(max_lag, max(1, n // 4))
    C = np.empty(safe_max_lag + 1)
    C[0] = 1.0
    for lag in range(1, safe_max_lag + 1):
        C[lag] = np.mean(x[:n - lag] * x[lag:]) / var
    tau = 0.5
    for M in range(1, len(C)):
        tau += C[M]
        if M >= 5 * tau:
            return float(tau)
    return float(tau)  # 수렴 안 했으면 lower bound


def measure_dominant_period_fft(signal_1d):
    """입력 신호의 지배적 주기(period, 스윕 단위)를 FFT로 측정.

    measure_signal_tau()와 상호보완적인 관계: Sokal의 적분 자기상관시간은 "느슨하게
    상관된" 신호를 가정하므로 완벽한 교대 신호(dwell_sweeps=1)처럼 반상관이 극단적인
    경우 음수 같은 물리적으로 무의미한 값을 낸다. 반면 FFT 기반 주기 측정은 신호가
    저주파(느리게 변함)든 고주파(빠르게 교대함)든 상관없이 동일한 방식으로 잘 작동한다
    -- dwell_sweeps=1(사각파, 최고 주파수)도 dwell_sweeps=40(느린 신호, 낮은 주파수)도
    같은 척도(주기, 스윕 단위)로 직접 비교할 수 있게 해준다.

    DC 성분(주파수 0)을 제외한 파워스펙트럼에서 가장 강한 주파수 성분을 찾아
    period = 1/f (스윕 단위)를 반환한다.
    """
    x = np.asarray(signal_1d, dtype=float)
    x = x - x.mean()
    if np.all(np.abs(x) < 1e-15):
        return float("nan")
    n = len(x)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)  # 스윕당 1 샘플 기준, 주파수 단위: cycles/sweep
    spectrum[0] = 0.0  # DC 성분(평균) 제외
    peak_idx = np.argmax(spectrum)
    if freqs[peak_idx] <= 0:
        return float("nan")
    return float(1.0 / freqs[peak_idx])


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (x.mean() if n else float("nan")), float("nan"), n
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    return m, stats.t.ppf(0.975, df=n - 1) * sem, n


def run_sweep(dwells, seeds, L, T, K0, power, n_baseline, n_driven, n_therm, meas_stride,
              burn_in_frac, outdir, record_morphology=False, record_field_snapshots=False,
              snapshot_stride=None):
    """(dwell, seed) 조합이 끝날 때마다 즉시 CSV에 이어붙여 저장한다 -- 30번 실행 중
    하나가 오래 걸리는 배치 작업에서, 전체가 다 끝나야만 파일이 생기는 방식은 중간에
    뭔가 잘못되면(정전, 실수로 종료 등) 그 시점까지의 결과가 전부 날아가는 위험이 있어서
    고쳤다. 이미 완료된 조합은 CSV에서 자동으로 읽어와 건너뛰므로(resume), 중간에
    멈췄다가 다시 실행해도 처음부터 다시 돌지 않는다.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "dwell_sweep_summary.csv")

    if os.path.exists(path):
        done_df = pd.read_csv(path)
        if "dominant_period_fft" not in done_df.columns:
            # 예전(FFT 주기 측정 추가 전) 결과를 이어서 쓰는 경우 -- 이미 완료된 행들에
            # 이 컬럼만 채워 넣는다. 신호 자체는 seed로부터 결정론적으로 재생성되므로
            # (비싼 MC 시뮬레이션은 다시 안 돌리고) 신호만 다시 만들어서 backfill한다.
            print("기존 CSV에 dominant_period_fft 컬럼이 없어서, 완료된 행들에 대해 "
                  "신호만 재생성하여 채워 넣습니다 (시뮬레이션 재실행 없음)...")
            periods = []
            for _, r in done_df.iterrows():
                rng = np.random.default_rng(int(r["seed"]))
                sig = signals.gen_random_telegraph(n_driven, power, int(r["dwell_sweeps"]), rng)
                periods.append(measure_dominant_period_fft(sig))
            done_df["dominant_period_fft"] = periods
            done_df.to_csv(path, index=False)
            print(f"  backfill 완료: {path} ({len(done_df)}행)")
        if "signal_type" not in done_df.columns:
            print("기존 CSV에 signal_type 컬럼이 없어서 채워 넣습니다 "
                  "(dwell=1 -> 결정론적 사각파, 나머지 -> random telegraph)...")
            done_df["signal_type"] = done_df["dwell_sweeps"].apply(
                lambda d: "deterministic period-2 square wave" if d == 1 else "random telegraph")
            done_df.to_csv(path, index=False)
            print(f"  backfill 완료: {path} ({len(done_df)}행)")
        done_set = set(zip(done_df["dwell_sweeps"], done_df["seed"]))
        print(f"기존 결과 발견: {path} ({len(done_df)}행) -- 이미 끝난 조합은 건너뜁니다.")
    else:
        done_set = set()

    all_combos = [(d, s) for d in dwells for s in seeds]
    remaining = [(d, s) for d, s in all_combos if (d, s) not in done_set]
    print(f"전체 {len(all_combos)}개 조합 중 {len(all_combos) - len(remaining)}개는 이미 "
          f"완료, {len(remaining)}개 남음.")

    for dwell, seed in remaining:
        print(f"\n--- dwell_sweeps={dwell}, seed={seed} "
              f"({all_combos.index((dwell, seed)) + 1}/{len(all_combos)}) ---")
        rec, driven_start_idx, driven_signal, delta_K = run_one_case(
            "telegraph", L, T, K0, power, n_baseline, n_driven, n_therm, meas_stride,
            seed=seed, dwell_sweeps=dwell, record_morphology=record_morphology,
            record_field_snapshots=record_field_snapshots, snapshot_stride=snapshot_stride)
        summary = summarize_case("telegraph", rec, driven_start_idx, driven_signal,
                                  delta_K, meas_stride, period_sweeps=40,
                                  burn_in_frac=burn_in_frac,
                                  signal_timescale_sweeps=2 * dwell,
                                  run_metadata_extra={
                                      "seed": seed, "L": L, "T": T, "K0": K0,
                                      "power": power, "meas_stride": meas_stride,
                                      "dwell_sweeps": dwell,
                                  })
        if record_morphology or record_field_snapshots:
            from run_stage1_experiment import save_case_timeseries
            sub_outdir = os.path.join(outdir, f"dwell{dwell}_seed{seed}")
            os.makedirs(sub_outdir, exist_ok=True)
            save_case_timeseries("telegraph", rec, sub_outdir)
        tau_signal = measure_signal_tau(driven_signal)
        dominant_period = measure_dominant_period_fft(driven_signal)
        # 채찍GPT 지적: dwell_sweeps=1은 매 스윕마다 무조건 부호가 바뀌므로 확률적
        # 요소가 전혀 없는 결정론적 period-2 사각파다 -- "이상하게 나온 telegraph"로
        # 취급하지 않고, 애초에 다른 신호 종류로 명시적으로 분류한다. dwell>=2부터는
        # 진짜 확률과정(random telegraph)이다.
        signal_type = "deterministic period-2 square wave" if dwell == 1 else "random telegraph"
        row = pd.DataFrame([{
            "dwell_sweeps": dwell,
            "seed": seed,
            "signal_type": signal_type,
            "tau_signal": tau_signal,
            "dominant_period_fft": dominant_period,
            "input_entropy_bits": summary["input_shannon_entropy_bits"],
            "net_TE_input_birth": summary["net_TE_input_birth"],
            "delta_birth": summary["delta_birth"],
            "mean_absorbed_energy": summary["mean_absorbed_energy"],
            "effect_size_energy": summary["effect_size_energy"],
            "effect_size_drop": summary["effect_size_drop"],
            "effect_size_birth": summary["effect_size_birth"],
            "effect_size_info": summary["effect_size_info"],
            "effect_size_helicity": summary["effect_size_helicity"],
            "effect_size_birth_label": summary["effect_size_birth_label"],
            "run_id": summary["run_id"],
            "script_version": summary["script_version"],
            "git_hash": summary["git_hash"],
            "run_timestamp": summary["run_timestamp"],
            "L": summary.get("L"), "T": summary.get("T"), "K0": summary.get("K0"),
            "power": summary.get("power"),
        }])
        if os.path.exists(path):
            existing_cols = pd.read_csv(path, nrows=0).columns
            row = row.reindex(columns=existing_cols)
        # 매 조합이 끝날 때마다 즉시 파일에 이어쓰기 (헤더는 파일이 없을 때만)
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"  -> {path}에 저장 완료 (지금까지 {all_combos.index((dwell, seed)) + 1}/"
              f"{len(all_combos)}개 조합)")

    df = pd.read_csv(path)
    print(f"\n전체 완료: {path} ({len(df)}행 = {len(dwells)}개 dwell값 x {len(seeds)}개 시드)")
    return df


def print_table(df, dwells):
    print("\n" + "=" * 70)
    print("DWELL-TIME 스윕 결과 (시드 평균 +/- 95% CI)")
    print("=" * 70)
    for dwell in dwells:
        sub = df[df["dwell_sweeps"] == dwell]
        signal_type = sub["signal_type"].iloc[0] if "signal_type" in sub.columns else "?"
        tau_m, tau_ci, _ = mean_ci95(sub["tau_signal"])
        period_m, period_ci, _ = mean_ci95(sub["dominant_period_fft"])
        h_m, h_ci, _ = mean_ci95(sub["input_entropy_bits"])
        te_m, te_ci, n = mean_ci95(sub["net_TE_input_birth"])
        print(f"  dwell={dwell:4d}  [{signal_type}]")
        if signal_type == "deterministic period-2 square wave":
            print(f"           (확률과정이 아니므로 tau_signal/Sokal은 적용 대상이 아님 "
                  f"-- period_fft만 참고)")
        else:
            print(f"           tau_signal={tau_m:6.2f}+/-{tau_ci:.2f}")
        print(f"           period_fft={period_m:6.2f}+/-{period_ci:.2f} sweeps  "
              f"entropy={h_m:.3f}+/-{h_ci:.3f} bits  "
              f"net_TE={te_m:+.4f}+/-{te_ci:.4f}  (n_seeds={n})")

    entropies = [mean_ci95(df[df["dwell_sweeps"] == d]["input_entropy_bits"])[0] for d in dwells]
    print(f"\n  엔트로피 범위: {min(entropies):.3f} ~ {max(entropies):.3f} bits "
          f"({'거의 고정됨 -- 의도한 대로 correlation time만 바뀐 것' if max(entropies)-min(entropies) < 0.1 else '생각보다 많이 변함 -- 엔트로피도 같이 바뀌고 있어서 해석에 주의 필요'})")

    if 1 in dwells:
        print(f"\n  dwell=1은 random telegraph 계열이 아니라 결정론적 period-2 사각파입니다 "
              f"(signal_type 컬럼 참고). 이 지점과 sine(period=40) 사이, 그리고 그 너머의 "
              f"주기들을 제대로 스윕하려면 square_wave_sweep.py를 쓰세요 -- "
              f"'python square_wave_sweep.py --periods 2,4,8,16,32,64,128 ...'")




def make_plots(df, dwells, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau_means, tau_cis = [], []
    period_means, period_cis = [], []
    h_means, h_cis = [], []
    te_means, te_cis = [], []
    for d in dwells:
        sub = df[df["dwell_sweeps"] == d]
        tm, tc, _ = mean_ci95(sub["tau_signal"])
        pm, pc, _ = mean_ci95(sub["dominant_period_fft"])
        hm, hc, _ = mean_ci95(sub["input_entropy_bits"])
        tem, tec, _ = mean_ci95(sub["net_TE_input_birth"])
        tau_means.append(tm); tau_cis.append(tc)
        period_means.append(pm); period_cis.append(pc)
        h_means.append(hm); h_cis.append(hc)
        te_means.append(tem); te_cis.append(tec)

    neg_tau = [t < 0 for t in tau_means]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # tau_signal이 음수인 dwell값은 다른 색(회색 X)으로 표시해서, Sokal 공식이 깨진
    # 지점이라는 걸 그림에서도 바로 알아볼 수 있게 한다.
    colors = ["tab:gray" if neg else "tab:purple" for neg in neg_tau]
    axes[0].errorbar(dwells, tau_means, yerr=tau_cis, fmt="o", capsize=4, color="tab:purple",
                      ecolor="tab:purple", zorder=1)
    axes[0].plot(dwells, tau_means, "-", color="tab:purple", alpha=0.4, zorder=0)
    for d, tm, neg in zip(dwells, tau_means, neg_tau):
        if neg:
            axes[0].scatter([d], [tm], color="tab:gray", marker="x", s=100, zorder=2,
                            label="Sokal formula breaks down (negative)" if d == dwells[neg_tau.index(True)] else None)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xlabel("dwell_sweeps (generator parameter)")
    axes[0].set_ylabel("measured signal tau_auto (Sokal, sweeps)")
    axes[0].set_title("tau_auto (Sokal) -- breaks down (negative)\nfor near-perfect alternation")
    if any(neg_tau):
        axes[0].legend(fontsize=7)

    axes[1].errorbar(dwells, h_means, yerr=h_cis, fmt="o-", capsize=4, color="tab:gray")
    axes[1].set_xlabel("dwell_sweeps")
    axes[1].set_ylabel("input Shannon entropy (bits)")
    axes[1].set_title("Sanity check: entropy should stay\napprox. flat across dwell values")
    axes[1].set_ylim(0, 2)

    axes[2].errorbar(dwells, te_means, yerr=te_cis, fmt="o-", capsize=4, color="tab:red")
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_xlabel("dwell_sweeps")
    axes[2].set_ylabel("net TE(input -> Birth)")
    axes[2].set_title("Main result: TE vs dwell_sweeps\n(at ~fixed input entropy)")

    fig.tight_layout()
    path = os.path.join(outdir, "dwell_sweep_TE_vs_correlation_time.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"저장: {path}")

    # tau_signal (Sokal) 대 TE -- 참고용. dwell=1처럼 tau_signal이 음수로 깨지는 지점은
    # 이 그림에서 별도 마커로 표시한다.
    fig2, ax = plt.subplots(figsize=(6, 5))
    for tm, tec_, tem, d, neg in zip(tau_means, tau_cis, te_means, dwells, neg_tau):
        color = "tab:gray" if neg else "tab:red"
        marker = "x" if neg else "o"
        ax.errorbar([tm], [tem], xerr=[tec_], yerr=None, fmt=marker, capsize=4, color=color)
        ax.annotate(f"dwell={d}" + (" (Sokal broken)" if neg else ""), (tm, tem),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("measured signal tau_auto (Sokal, sweeps) -- unreliable where marked (x)")
    ax.set_ylabel("net TE(input -> Birth)")
    ax.set_title("Net TE vs Sokal tau_auto (reference only)")
    fig2.tight_layout()
    path2 = os.path.join(outdir, "dwell_sweep_TE_vs_tau_signal.png")
    fig2.savefig(path2, dpi=130)
    plt.close(fig2)
    print(f"저장: {path2}")

    # ** 메인 결과 **: FFT 기반 주기(period) 대 TE -- 반상관 신호(dwell=1)에서도 깨지지
    # 않으므로, 모든 dwell값을 하나의 일관된 축(주기, 스윕 단위)으로 직접 비교할 수 있다.
    fig3, ax3 = plt.subplots(figsize=(6.5, 5.5))
    ax3.errorbar(period_means, te_means, xerr=period_cis, yerr=te_cis, fmt="o", capsize=4,
                 color="tab:blue")
    for pm, tem, d in zip(period_means, te_means, dwells):
        ax3.annotate(f"dwell={d}", (pm, tem), textcoords="offset points", xytext=(6, 6),
                     fontsize=8)
    ax3.axhline(0, color="black", lw=0.6)
    ax3.set_xlabel("dominant period (FFT, sweeps)")
    ax3.set_ylabel("net TE(input -> Birth)")
    ax3.set_title("MAIN RESULT: Net TE vs input's dominant period\n"
                  "(FFT-based -- valid even for the dwell=1 square wave)")
    fig3.tight_layout()
    path3 = os.path.join(outdir, "dwell_sweep_TE_vs_dominant_period.png")
    fig3.savefig(path3, dpi=130)
    plt.close(fig3)
    print(f"저장: {path3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwells", type=str, default="1,2,5,10,20,40",
                     help="콤마로 구분한 dwell_sweeps 값 목록")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                     help="콤마로 구분한 시드 목록 (각 dwell값마다 이 시드들로 반복)")
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--T", type=float, default=0.9)
    ap.add_argument("--K0", type=float, default=0.7)
    ap.add_argument("--power", type=float, default=0.01)
    ap.add_argument("--n-baseline", type=int, default=5000)
    ap.add_argument("--n-driven", type=int, default=30000)
    ap.add_argument("--n-therm", type=int, default=45)
    ap.add_argument("--meas-stride", type=int, default=5)
    ap.add_argument("--burn-in-frac", type=float, default=0.0006)
    ap.add_argument("--outdir", type=str, default="dwell_sweep_results")
    ap.add_argument("--record-morphology", action="store_true",
                     help="Paper 4용: 정보 필드 형태 스칼라 저장")
    ap.add_argument("--record-field-snapshots", action="store_true",
                     help="Paper 4용: 정보 필드 원본을 .npz로 저장 (저장공간 필요)")
    ap.add_argument("--snapshot-stride", type=int, default=None)
    args = ap.parse_args()

    dwells = [int(d) for d in args.dwells.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"dwell_sweeps 값: {dwells}")
    print(f"시드: {seeds}")
    print(f"총 실행 횟수: {len(dwells) * len(seeds)}")

    df = run_sweep(dwells, seeds, args.L, args.T, args.K0, args.power, args.n_baseline,
                    args.n_driven, args.n_therm, args.meas_stride, args.burn_in_frac,
                    args.outdir, record_morphology=args.record_morphology,
                    record_field_snapshots=args.record_field_snapshots,
                    snapshot_stride=args.snapshot_stride)
    print_table(df, dwells)
    make_plots(df, dwells, args.outdir)


if __name__ == "__main__":
    main()
