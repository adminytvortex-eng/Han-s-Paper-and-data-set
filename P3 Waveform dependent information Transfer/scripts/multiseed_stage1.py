"""
Multi-seed reproducibility check for Paper 3, Stage 1.

Runs the full 3-case protocol across several seeds (reusing run_stage1_experiment.py's
already-tested per-case functions, not reimplementing them) and produces exactly the four
figures requested after the pilot review:

  Figure A: Delta-Birth (raw, not the ratio -- efficiency is sensitive to a small
            denominator, Delta-Birth itself is the most primitive, least-processed
            observable and should be looked at first)
  Figure B: Absorbed energy
  Figure C: Efficiency (Delta-Birth / Delta-E_abs)
  Figure D: Net transfer entropy (input -> Birth)

Each as mean +/- 95% CI across seeds, one point/bar per case (Sine / White / Telegraph).

The single most important thing to look at first: does White's Delta-Birth stay NEGATIVE
across seeds, and does Sine's net TE(->Birth) stay large? If both hold up, this is no longer
a single-seed curiosity.

Usage:
    python multiseed_stage1.py --seeds 0,1,2,3,4,5,6 --L 48 --T 0.9 --K0 0.7 --power 0.01 \\
        --n-baseline 5000 --n-driven 30000 --n-therm 45 --meas-stride 5 \\
        --burn-in-frac 0.0006 --outdir multiseed_results
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from run_stage1_experiment import (run_one_case, summarize_case, save_case_timeseries,
                                    signal_timescale_for_case, CASES)


def run_all_seeds(seeds, L, T, K0, power, n_baseline, n_driven, n_therm, meas_stride,
                   period_sweeps, dwell_sweeps, burn_in_frac, outdir_root, save_timeseries=True,
                   record_morphology=False, record_field_snapshots=False,
                   snapshot_stride=None, cases=None):
    """(seed, case) 조합이 끝날 때마다 즉시 multiseed_summary.csv에 이어붙여 저장한다.
    이전 버전은 전체 시드 x 케이스가 다 끝나야만 요약 CSV가 생겼는데, 시드 개수가 많은
    배치 작업에서 중간에 멈추면 그때까지의 결과가 전부 날아가는 문제가 있었다 (원본
    시계열 CSV는 케이스별로 즉시 저장되지만, 집계된 요약은 아니었음). 이미 완료된
    (seed, case) 조합은 자동으로 건너뛰므로(resume), 같은 명령으로 다시 실행해도 처음부터
    다시 돌지 않는다.

    cases: None이면 CASES(sine/white/telegraph) 전부. 특정 케이스만 다시 돌리고 싶을 때
    (예: sine만 다른 period_sweeps로 재마이닝) 리스트로 지정한다 -- 이미 HIGH 등급으로
    확보된 case를 불필요하게 다시 안 돌기 위함.
    """
    cases = cases if cases is not None else CASES
    os.makedirs(outdir_root, exist_ok=True)
    path = os.path.join(outdir_root, "multiseed_summary.csv")

    if os.path.exists(path):
        done_df = pd.read_csv(path)
        done_set = set(zip(done_df["seed"], done_df["case"]))
        print(f"기존 결과 발견: {path} ({len(done_df)}행) -- 이미 끝난 조합은 건너뜁니다.")
    else:
        done_set = set()

    all_combos = [(seed, case) for seed in seeds for case in cases]
    remaining = [(seed, case) for seed, case in all_combos if (seed, case) not in done_set]
    print(f"전체 {len(all_combos)}개 조합(시드 x 케이스) 중 "
          f"{len(all_combos) - len(remaining)}개는 이미 완료, {len(remaining)}개 남음.")

    for seed, case in remaining:
        idx = all_combos.index((seed, case)) + 1
        print(f"\n{'#' * 70}\n# seed={seed}, case={case}  ({idx}/{len(all_combos)})\n{'#' * 70}")
        seed_outdir = os.path.join(outdir_root, f"seed{seed}")
        os.makedirs(seed_outdir, exist_ok=True)

        rec, driven_start_idx, driven_signal, delta_K = run_one_case(
            case, L, T, K0, power, n_baseline, n_driven, n_therm, meas_stride, seed=seed,
            period_sweeps=period_sweeps, dwell_sweeps=dwell_sweeps,
            record_morphology=record_morphology,
            record_field_snapshots=record_field_snapshots,
            snapshot_stride=snapshot_stride)
        if save_timeseries:
            save_case_timeseries(case, rec, seed_outdir)
        summary = summarize_case(case, rec, driven_start_idx, driven_signal, delta_K,
                                  meas_stride, period_sweeps, burn_in_frac=burn_in_frac,
                                  signal_timescale_sweeps=signal_timescale_for_case(
                                      case, period_sweeps, dwell_sweeps),
                                  run_metadata_extra={
                                      "seed": seed, "L": L, "T": T, "K0": K0,
                                      "power": power, "meas_stride": meas_stride,
                                      "period_sweeps": period_sweeps,
                                      "dwell_sweeps": dwell_sweeps,
                                  })
        summary["seed"] = seed

        row = pd.DataFrame([summary])
        if os.path.exists(path):
            # 방어적 조치: 헤더가 이미 있으면 그 열 순서에 강제로 맞춘다 (reindex).
            # summarize_case가 케이스마다 다른 순서로 키를 만들면 위치 기반인 CSV append가
            # 조용히 잘못된 열에 값을 써버리는 사고가 실제로 있었다 (seed 컬럼이 NaN으로
            #깨지는 형태로 나타났음) -- 근본 원인은 고쳤지만, 앞으로 같은 종류의 실수가
            # 또 나도 조용히 깨지지 않도록 여기서도 한 번 더 강제한다.
            existing_cols = pd.read_csv(path, nrows=0).columns
            missing = set(existing_cols) - set(row.columns)
            extra = set(row.columns) - set(existing_cols)
            if missing or extra:
                raise RuntimeError(
                    f"컬럼 집합이 기존 CSV와 다릅니다 (missing={missing}, extra={extra}) -- "
                    f"summarize_case()가 케이스마다 다른 키를 만들고 있다는 뜻이니, 코드를 "
                    f"고치기 전에는 이어쓰기를 하면 안 됩니다.")
            row = row.reindex(columns=existing_cols)
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"  -> {path}에 저장 완료 (지금까지 {idx}/{len(all_combos)}개 조합)")

    df = pd.read_csv(path)
    print(f"\n전체 완료: {path} ({len(df)}행 = {len(seeds)}개 시드 x {len(CASES)}개 케이스)")
    return df


def mean_ci95(x, min_n=2):
    """Mean and half-width of a 95% CI using Student's t (appropriate for small seed counts --
    normal-approximation CIs are too narrow with only a handful of seeds).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_n:
        return (x.mean() if n else float("nan")), float("nan"), n
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return m, t_crit * sem, n


def aggregate(df, metrics):
    """Returns {metric: {case: (mean, ci95_halfwidth, n)}}."""
    out = {}
    for metric in metrics:
        out[metric] = {}
        for case in CASES:
            vals = df.loc[df["case"] == case, metric]
            out[metric][case] = mean_ci95(vals)
    return out


def print_summary_table(agg, metrics, labels):
    print("\n" + "=" * 70)
    print("SEED-AGGREGATED RESULTS (mean +/- 95% CI)")
    print("=" * 70)
    for metric, label in zip(metrics, labels):
        print(f"\n{label}:")
        for case in CASES:
            m, ci, n = agg[metric][case]
            sign_flag = ""
            if metric == "delta_birth" and np.isfinite(m):
                sign_flag = "  <-- NEGATIVE" if m < 0 else ""
            print(f"  {case:10s}  {m:+.5g}  +/- {ci:.3g}  (n_seeds={n}){sign_flag}")


def make_figures(agg, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("delta_birth", "Figure A: Delta-Birth (raw)", "delta_birth.png"),
        ("mean_absorbed_energy", "Figure B: Absorbed energy", "absorbed_energy.png"),
        ("efficiency_birth_per_absorbedE", "Figure C: Efficiency (Delta-Birth / Delta-E_abs)",
         "efficiency.png"),
        ("net_TE_input_birth", "Figure D: Net transfer entropy (input -> Birth)",
         "net_TE_birth.png"),
    ]

    for metric, title, fname in specs:
        fig, ax = plt.subplots(figsize=(5, 4))
        means = [agg[metric][c][0] for c in CASES]
        cis = [agg[metric][c][1] for c in CASES]
        colors = ["tab:blue" if m >= 0 or not np.isfinite(m) else "tab:red" for m in means]
        ax.bar(CASES, means, yerr=cis, capsize=5, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(metric)
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  saved {path}")

    # combined 2x2 view for a quick single-glance check
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (metric, title, _) in zip(axes.flat, specs):
        means = [agg[metric][c][0] for c in CASES]
        cis = [agg[metric][c][1] for c in CASES]
        colors = ["tab:blue" if m >= 0 or not np.isfinite(m) else "tab:red" for m in means]
        ax.bar(CASES, means, yerr=cis, capsize=5, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    combined_path = os.path.join(outdir, "all_four_figures.png")
    fig.savefig(combined_path, dpi=130)
    plt.close(fig)
    print(f"  saved {combined_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6",
                     help="comma-separated seed list, e.g. '0,1,2,3,4,5,6' for 7 seeds")
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--T", type=float, default=0.9)
    ap.add_argument("--K0", type=float, default=0.7)
    ap.add_argument("--power", type=float, default=0.01)
    ap.add_argument("--n-baseline", type=int, default=5000)
    ap.add_argument("--n-driven", type=int, default=30000)
    ap.add_argument("--n-therm", type=int, default=45)
    ap.add_argument("--meas-stride", type=int, default=5)
    ap.add_argument("--period-sweeps", type=int, default=40)
    ap.add_argument("--dwell-sweeps", type=int, default=20)
    ap.add_argument("--burn-in-frac", type=float, default=0.0006)
    ap.add_argument("--outdir", type=str, default="multiseed_results")
    ap.add_argument("--no-save-timeseries", action="store_true",
                     help="skip saving per-seed per-case timeseries CSVs (saves disk space "
                          "for large multi-seed runs; the aggregated summary is unaffected)")
    ap.add_argument("--record-morphology", action="store_true",
                     help="Paper 4용: 정보 필드 형태 스칼라(mean/max/std/skew/...) 저장")
    ap.add_argument("--record-field-snapshots", action="store_true",
                     help="Paper 4용: 정보 필드 원본을 .npz로 저장 (저장공간 필요)")
    ap.add_argument("--snapshot-stride", type=int, default=None,
                     help="필드 스냅샷 저장 간격(스윕). 기본값은 meas_stride*10")
    ap.add_argument("--cases", type=str, default=None,
                     help="콤마로 구분한 케이스 목록 (기본: sine,white,telegraph 전부). "
                          "예: --cases sine 이면 sine만 다시 돈다 -- 이미 HIGH 등급으로 "
                          "확보된 케이스를 불필요하게 재실행하지 않기 위함")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Running {len(seeds)} seeds: {seeds}")
    cases = [c.strip() for c in args.cases.split(",")] if args.cases else None
    if cases:
        print(f"지정된 케이스만 실행: {cases}")

    df = run_all_seeds(seeds, args.L, args.T, args.K0, args.power, args.n_baseline,
                        args.n_driven, args.n_therm, args.meas_stride, args.period_sweeps,
                        args.dwell_sweeps, args.burn_in_frac, args.outdir,
                        save_timeseries=not args.no_save_timeseries,
                        record_morphology=args.record_morphology,
                        record_field_snapshots=args.record_field_snapshots,
                        snapshot_stride=args.snapshot_stride, cases=cases)

    metrics = ["delta_birth", "mean_absorbed_energy", "efficiency_birth_per_absorbedE",
               "net_TE_input_birth"]
    labels = ["Delta-Birth (raw)", "Absorbed energy", "Efficiency (Birth/AbsE)",
              "Net TE(input->Birth)"]
    agg = aggregate(df, metrics)
    print_summary_table(agg, metrics, labels)

    print(f"\nSaving figures to {args.outdir}/...")
    make_figures(agg, args.outdir)

    m_white, ci_white, n_white = agg["delta_birth"]["white"]
    m_sine_te, ci_sine_te, n_sine_te = agg["net_TE_input_birth"]["sine"]
    print("\n" + "=" * 70)
    print("HEADLINE CHECK (the two things the pilot flagged as most interesting)")
    print("=" * 70)

    def describe(m, ci):
        if not np.isfinite(m) or not np.isfinite(ci):
            return "not enough seeds to compute a CI yet"
        if m + ci < 0:
            return "CONFIRMED NEGATIVE (entire 95% CI is below zero)"
        if m - ci > 0:
            return "CONFIRMED POSITIVE (entire 95% CI is above zero)"
        return "CI crosses zero -- not yet distinguishable from no effect, need more seeds"

    print(f"  White Delta-Birth: {m_white:+.5g} +/- {ci_white:.3g} -> {describe(m_white, ci_white)}")
    print(f"  Sine net TE(->Birth): {m_sine_te:+.5g} +/- {ci_sine_te:.3g} -> "
          f"{describe(m_sine_te, ci_sine_te)}")


if __name__ == "__main__":
    main()
