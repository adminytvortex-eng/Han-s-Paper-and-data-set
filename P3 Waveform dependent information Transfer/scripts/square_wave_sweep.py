"""
결정론적 사각파(square wave)의 period_sweeps를 스윕하는 실험.

채찍GPT 피드백 반영: dwell_time_sweep.py에서 dwell_sweeps=1은 사실 확률과정(random
telegraph)이 아니라 완전히 결정론적인 period-2 사각파였다 (매 스윕마다 무조건 부호가
바뀌므로). 이걸 "이상한 telegraph 결과"로 억지로 끼워맞추지 말고, 애초에 signals.py에
`gen_square_wave`라는 별도의 신호 계열로 분리했다. 이 스크립트는 그 사각파의 주기를
자유롭게 스윕해서 (period=2,4,8,16,32,64,...), dwell=1(period=2)과 sine(period=40)
사이에 비어 있던 구간을 채운다.

이렇게 하면 입력 신호를 특정짓는 세 축 -- entropy(H), correlation time(τ),
dominant frequency(ω) -- 중 ω축을 독립적으로 스윕할 수 있게 된다:
  - Shannon entropy는 사각파에서 항상 정확히 1비트로 고정 (telegraph와 동일한 이유:
    +-A 두 값만 가짐)
  - 사각파는 결정론적이므로 correlation time이라는 개념 자체가 telegraph와 다르게
    적용됨 (확률적 dwell이 아니라 정확한 period)
  - period_sweeps 하나로 dominant frequency(omega=2*pi/period)를 직접 제어

측정하는 것:
  1. net TE(input -> Birth) -- period에 따라 어떻게 변하는지
  2. phase lag phi(omega) -- 사각파도 sine처럼 결정론적으로 주기적이므로, phase_lag가
     의미 있게 정의된다 (cross-correlation tau가 아니라)
  3. input Shannon entropy -- 항상 1비트 근처로 고정되는지 확인

사용법:
    python square_wave_sweep.py --periods 2,4,8,16,32,64,128 --seeds 0,1,2,3,4 \\
        --L 48 --T 0.9 --K0 0.7 --power 0.01 --n-baseline 5000 --n-driven 30000 \\
        --n-therm 45 --meas-stride 5 --burn-in-frac 0.0006 --outdir square_sweep_results
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from run_stage1_experiment import run_one_case, summarize_case


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (x.mean() if n else float("nan")), float("nan"), n
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    return m, stats.t.ppf(0.975, df=n - 1) * sem, n


def run_sweep(periods, seeds, L, T, K0, power, n_baseline, n_driven, n_therm, meas_stride,
              burn_in_frac, outdir, record_morphology=False, record_field_snapshots=False,
              snapshot_stride=None):
    """(period, seed) 조합이 끝날 때마다 즉시 CSV에 이어붙여 저장 + resume 지원
    (dwell_time_sweep.py / multiseed_stage1.py와 동일한 패턴).

    record_morphology/record_field_snapshots가 켜져 있으면, 요약 스칼라만 뽑는 게 아니라
    run_stage1_experiment.save_case_timeseries()를 통해 각 (period, seed)별 전체
    timeseries(+형태 스칼라/필드 스냅샷)를 outdir/period{P}_seed{S}/ 밑에 따로 저장한다
    (Paper 4용 원본 데이터 보존).
    """
    from run_stage1_experiment import save_case_timeseries

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "square_sweep_summary.csv")

    if os.path.exists(path):
        done_df = pd.read_csv(path)
        done_set = set(zip(done_df["period_sweeps"], done_df["seed"]))
        print(f"기존 결과 발견: {path} ({len(done_df)}행) -- 이미 끝난 조합은 건너뜁니다.")
    else:
        done_set = set()

    all_combos = [(p, s) for p in periods for s in seeds]
    remaining = [(p, s) for p, s in all_combos if (p, s) not in done_set]
    print(f"전체 {len(all_combos)}개 조합 중 {len(all_combos) - len(remaining)}개는 이미 "
          f"완료, {len(remaining)}개 남음.")

    for period, seed in remaining:
        idx = all_combos.index((period, seed)) + 1
        print(f"\n--- period_sweeps={period}, seed={seed} ({idx}/{len(all_combos)}) ---")
        rec, driven_start_idx, driven_signal, delta_K = run_one_case(
            "square", L, T, K0, power, n_baseline, n_driven, n_therm, meas_stride,
            seed=seed, period_sweeps=period, record_morphology=record_morphology,
            record_field_snapshots=record_field_snapshots, snapshot_stride=snapshot_stride)
        summary = summarize_case("square", rec, driven_start_idx, driven_signal,
                                  delta_K, meas_stride, period_sweeps=period,
                                  burn_in_frac=burn_in_frac,
                                  run_metadata_extra={
                                      "seed": seed, "L": L, "T": T, "K0": K0,
                                      "power": power, "meas_stride": meas_stride,
                                      "period_sweeps": period,
                                  })

        if record_morphology or record_field_snapshots:
            sub_outdir = os.path.join(outdir, f"period{period}_seed{seed}")
            os.makedirs(sub_outdir, exist_ok=True)
            save_case_timeseries("square", rec, sub_outdir)

        row = pd.DataFrame([{
            "period_sweeps": period,
            "seed": seed,
            "input_entropy_bits": summary["input_shannon_entropy_bits"],
            "net_TE_input_birth": summary["net_TE_input_birth"],
            "te_reliable": summary["te_reliable"],
            "te_reliability": summary["te_reliability"],
            "period_to_stride_ratio": summary["signal_period_to_stride_ratio"],
            "te_n_cycles": summary["te_n_cycles"],
            "phase_lag_birth_rad": summary["phase_lag_input_to_birth_rad"],
            "phase_lag_amp_birth": summary["phase_lag_amp_out_birth"],
            "delta_birth": summary["delta_birth"],
            "mean_absorbed_energy": summary["mean_absorbed_energy"],
            # 채찍GPT 권고: 효과크기(Hedges' g, baseline vs driven) -- p<0.05뿐 아니라
            # 효과가 실제로 얼마나 큰지도 같이 저장
            "effect_size_energy": summary["effect_size_energy"],
            "effect_size_drop": summary["effect_size_drop"],
            "effect_size_birth": summary["effect_size_birth"],
            "effect_size_info": summary["effect_size_info"],
            "effect_size_helicity": summary["effect_size_helicity"],
            "effect_size_birth_label": summary["effect_size_birth_label"],
            # run metadata (P4/P5 추적용)
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
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"  -> {path}에 저장 완료 (지금까지 {idx}/{len(all_combos)}개 조합)")

    df = pd.read_csv(path)
    print(f"\n전체 완료: {path} ({len(df)}행 = {len(periods)}개 period값 x {len(seeds)}개 시드)")
    return df


def print_table(df, periods, meas_stride):
    print("\n" + "=" * 70)
    print("SQUARE-WAVE PERIOD 스윕 결과 (시드 평균 +/- 95% CI)")
    print("=" * 70)
    for period in periods:
        sub = df[df["period_sweeps"] == period]
        h_m, h_ci, _ = mean_ci95(sub["input_entropy_bits"])
        te_m, te_ci, n = mean_ci95(sub["net_TE_input_birth"])
        phi_m, phi_ci, _ = mean_ci95(sub["phase_lag_birth_rad"])
        ratio = sub["period_to_stride_ratio"].iloc[0] if "period_to_stride_ratio" in sub else period / meas_stride
        n_cycles = sub["te_n_cycles"].iloc[0] if "te_n_cycles" in sub else float("nan")
        reliability = sub["te_reliability"].iloc[0] if "te_reliability" in sub.columns else "?"
        flag = "" if reliability == "HIGH" else f"  <-- {reliability} (samples/period={ratio:.1f}, n_cycles={n_cycles:.1f})"
        print(f"  period={period:5d}  entropy={h_m:.3f}+/-{h_ci:.3f} bits  "
              f"phase_lag={phi_m:+.3f}+/-{phi_ci:.3f} rad  (n_seeds={n})")
        print(f"           net_TE={te_m:+.4f}+/-{te_ci:.4f}{flag}")

    entropies = [mean_ci95(df[df["period_sweeps"] == p]["input_entropy_bits"])[0] for p in periods]
    print(f"\n  엔트로피 범위: {min(entropies):.3f} ~ {max(entropies):.3f} bits "
          f"({'거의 고정됨' if max(entropies) - min(entropies) < 0.1 else '많이 변함 -- 주의'})")

    if "te_reliability" in df.columns:
        not_high = [p for p in periods
                    if df[df["period_sweeps"] == p]["te_reliability"].iloc[0] != "HIGH"]
        if not_high:
            print(f"\n  나이퀴스트/사이클 경고: period={not_high} 는 HIGH 등급이 아닙니다 "
                  f"(자세한 기준은 protocol_check.py 참고). 이 period들의 net_TE는 본 "
                  f"분석/그래프에서 제외하고 phase_lag만 사용하세요.")


TIER_COLOR = {"HIGH": "tab:blue", "MEDIUM": "tab:orange", "LOW": "tab:gray"}
TIER_MARKER = {"HIGH": "o", "MEDIUM": "s", "LOW": "x"}


def make_plots(df, periods, outdir, meas_stride):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    te_means, te_cis, phi_means, phi_cis, tiers = [], [], [], [], []
    for p in periods:
        sub = df[df["period_sweeps"] == p]
        tm, tc, _ = mean_ci95(sub["net_TE_input_birth"])
        pm, pc, _ = mean_ci95(sub["phase_lag_birth_rad"])
        te_means.append(tm); te_cis.append(tc)
        phi_means.append(pm); phi_cis.append(pc)
        tier = sub["te_reliability"].iloc[0] if "te_reliability" in sub.columns else "HIGH"
        tiers.append(tier)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, xscale in zip(axes, ["linear", "log"]):
        # HIGH 등급만 실선으로 연결 (채찍GPT 권고: TE 그래프에서 신뢰 불가 지점은 제외
        # 하고 phase_lag/cross-correlation만 그쪽에서 쓴다). MEDIUM/LOW는 등급별로 다른
        # 색/마커로 표시해서 "왜 빠졌는지"가 그림만 봐도 보이게 한다.
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            idx = [i for i, t in enumerate(tiers) if t == tier]
            if not idx:
                continue
            p_sub = [periods[i] for i in idx]
            te_sub = [te_means[i] for i in idx]
            ci_sub = [te_cis[i] for i in idx]
            style = "-" if tier == "HIGH" else "none"
            ax.errorbar(p_sub, te_sub, yerr=ci_sub, fmt=TIER_MARKER[tier], linestyle=style,
                        capsize=4, color=TIER_COLOR[tier], label=f"{tier}")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xscale(xscale)
        ax.set_xlabel(f"square wave period (sweeps{', log scale' if xscale == 'log' else ''})")
        ax.set_ylabel("net TE(input -> Birth)")
        ax.legend(fontsize=7, title="TE reliability")
    axes[0].set_title("Linear x-axis")
    axes[1].set_title("Log x-axis (usually clearer)")

    fig.suptitle(f"Net TE vs square-wave period (meas_stride={meas_stride})\n"
                 f"Only HIGH (blue, connected) should be used for TE conclusions -- "
                 f"see protocol_check.py")
    fig.tight_layout()
    path = os.path.join(outdir, "square_sweep_TE_vs_period.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"저장: {path}")

    # phase_lag는 나이퀴스트 문제에 훨씬 덜 민감하므로, 모든 period를 포함해서 그린다.
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.errorbar(periods, phi_means, yerr=phi_cis, fmt="o-", capsize=4, color="tab:green")
    ax2.set_xscale("log")
    ax2.set_xlabel("square wave period (sweeps, log scale)")
    ax2.set_ylabel("phase lag (rad)")
    ax2.set_title("Phase lag vs period (robust to the TE aliasing issue above)")
    fig2.tight_layout()
    path2 = os.path.join(outdir, "square_sweep_phase_lag_vs_period.png")
    fig2.savefig(path2, dpi=130)
    plt.close(fig2)
    print(f"저장: {path2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=str, default="2,4,8,16,32,64,128",
                     help="콤마로 구분한 square wave period_sweeps 값 목록")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--T", type=float, default=0.9)
    ap.add_argument("--K0", type=float, default=0.7)
    ap.add_argument("--power", type=float, default=0.01)
    ap.add_argument("--n-baseline", type=int, default=5000)
    ap.add_argument("--n-driven", type=int, default=30000)
    ap.add_argument("--n-therm", type=int, default=45)
    ap.add_argument("--meas-stride", type=int, default=5)
    ap.add_argument("--burn-in-frac", type=float, default=0.0006)
    ap.add_argument("--outdir", type=str, default="square_sweep_results")
    ap.add_argument("--record-morphology", action="store_true",
                     help="Paper 4용: 정보 필드 형태 스칼라 저장")
    ap.add_argument("--record-field-snapshots", action="store_true",
                     help="Paper 4용: 정보 필드 원본을 .npz로 저장 (저장공간 필요)")
    ap.add_argument("--snapshot-stride", type=int, default=None)
    args = ap.parse_args()

    periods = [int(p) for p in args.periods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"period_sweeps 값: {periods}")
    print(f"시드: {seeds}")
    print(f"총 실행 횟수: {len(periods) * len(seeds)}")

    from analyze_response import assess_te_reliability
    preflight = {p: assess_te_reliability(p, args.meas_stride, args.n_driven)["overall"]
                 for p in periods}
    not_high = [p for p, tier in preflight.items() if tier != "HIGH"]
    if not_high:
        print(f"\n*** 사전 점검 (protocol_check.py 기준): {preflight} ***")
        print(f"  HIGH가 아닌 period={not_high} 는 TE가 신뢰 불가능할 것으로 예상됩니다 "
              f"(실험 전 미리 경고).")
        print("  그래도 계속 진행합니다 -- 시뮬레이션 자체는 정상 수행되고, TE만 "
              "te_reliability로 등급이 표시되어 저장되며 그래프에서 자동으로 구분됩니다.\n")

    df = run_sweep(periods, seeds, args.L, args.T, args.K0, args.power, args.n_baseline,
                    args.n_driven, args.n_therm, args.meas_stride, args.burn_in_frac,
                    args.outdir, record_morphology=args.record_morphology,
                    record_field_snapshots=args.record_field_snapshots,
                    snapshot_stride=args.snapshot_stride)
    print_table(df, periods, args.meas_stride)
    make_plots(df, periods, args.outdir, args.meas_stride)


if __name__ == "__main__":
    main()
