"""
Top-level driver for Paper 3, Stage 1: "Does the lattice respond to energy, or to information
structure?" -- see the protocol document for the full experimental design and rationale.

Runs three input cases (Sin, White noise, Random telegraph) at matched input power, each as
a baseline (undriven) segment followed by a driven segment, and saves:
  - one CSV of raw time series per case          (stage1_<case>_timeseries.csv)
  - one summary CSV across all cases             (stage1_summary.csv)
  - one PNG figure per case (energy/birth/drop overlays + cross-correlation)
  - one PNG comparing the three cases directly   (stage1_comparison.png)

Usage:
    python run_stage1_experiment.py                      # default (small/fast) parameters
    python run_stage1_experiment.py --L 48 --n-driven 20000 --seeds 5

*** IMPORTANT -- run the linear-response check before trusting any result. ***
This script includes a `--check-linearity` flag that repeats the White noise case at half
and double the requested power and reports whether Delta-E scales roughly proportionally.
If it does not, the chosen `power` is too large (the drive is pushing the system out of the
linear-response regime) and every comparison across signal types becomes harder to interpret.
"""

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone

import numpy as np

from kagome_lattice import KagomeLattice
import signals
from driven_kagome_sim import run_driven_experiment, build_protocol_schedule
from analyze_response import (cross_correlation_lag, net_transfer_entropy,
                               phase_lag_at_frequency, omega_from_period,
                               check_nyquist_ratio, assess_te_reliability,
                               cohens_d, effect_size_label,
                               absorbed_energy, efficiency)

SCRIPT_VERSION = "paper3_stage1_v7"
# 이 버전 문자열은 파일들을 고칠 때마다 사람이 직접 올려주는 수동 태그다 -- git
# 커밋이 없는 환경(예: 그냥 폴더로 복사해서 쓰는 경우)에서도 "이 결과가 어느 코드
# 버전으로 만들어졌는지"를 CSV만 보고 알 수 있게 하기 위함.


def get_run_metadata():
    """git commit hash(있으면), 스크립트 버전, 마이닝을 실행한 시각을 반환한다 --
    P4/P5로 넘어갈 때 "이 데이터가 정확히 어느 코드로 만들어졌는지" 추적하기 위함
    (채찍GPT 권고). git 저장소가 아니어도(예: 그냥 폴더로 복사해서 쓰는 경우) 에러
    없이 "no-git"으로 채워진다.
    """
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        git_hash = "no-git"
    return {
        "run_id": uuid.uuid4().hex[:12],
        "script_version": SCRIPT_VERSION,
        "git_hash": git_hash,
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


CASES = ["sine", "white", "telegraph"]
OUTPUTS = ["energy", "drop", "birth", "info", "helicity"]  # priority order per protocol doc


def signal_timescale_for_case(case, period_sweeps, dwell_sweeps):
    """각 케이스에 맞는 나이퀴스트 체크용 시간척도를 반환한다.
    - sine/square: period_sweeps (결정론적 주기 그 자체)
    - telegraph: 2*dwell_sweeps (부호가 한 번 바뀌고 다시 돌아오는 대략적인 '주기')
    - white: 정의상 뚜렷한 주기가 없는 광대역 신호이므로, 이 나이퀴스트 체크(주기적 입력
      전용)를 적용할 대상이 아니다 -- 아주 큰 값을 줘서 항상 '신뢰 가능'으로 처리한다.
    """
    if case in ("sine", "square"):
        return period_sweeps
    elif case == "telegraph":
        return 2 * dwell_sweeps
    else:  # white -- 주기 개념이 없으므로 이 체크 대상이 아님
        return float("inf")


def make_signal(case, n_driven, power, rng, period_sweeps=40, dwell_sweeps=20):
    if case == "sine":
        sig = signals.gen_sine(n_driven, power, period_sweeps=period_sweeps, rng=rng)
    elif case == "white":
        sig = signals.gen_white_noise(n_driven, power, rng=rng)
    elif case == "telegraph":
        sig = signals.gen_random_telegraph(n_driven, power, dwell_sweeps=dwell_sweeps, rng=rng)
    elif case == "square":
        sig = signals.gen_square_wave(n_driven, power, period_sweeps=period_sweeps, rng=rng)
    else:
        raise ValueError(case)
    signals.check_power(sig, power, label=case)
    return sig


def run_one_case(case, L, T, K0, power, n_baseline, n_driven, n_therm, meas_stride, seed,
                  period_sweeps=40, dwell_sweeps=20, record_morphology=False,
                  record_field_snapshots=False, snapshot_stride=None):
    rng = np.random.default_rng(seed)
    driven_signal = make_signal(case, n_driven, power, rng, period_sweeps, dwell_sweeps)
    delta_K, driven_start_sweep = build_protocol_schedule(n_baseline, n_driven, driven_signal)

    print(f"\n=== Running case '{case}' (seed={seed}) ===")
    rec = run_driven_experiment(L, T, K0, delta_K, n_therm, meas_stride, rng,
                                 record_morphology=record_morphology,
                                 record_field_snapshots=record_field_snapshots,
                                 snapshot_stride=snapshot_stride)

    # index (in the *measurement*, not sweep, array) at which the driven segment begins
    driven_start_idx = int(np.searchsorted(rec["sweep"], driven_start_sweep))
    return rec, driven_start_idx, driven_signal, delta_K


def summarize_case(case, rec, driven_start_idx, driven_signal, delta_K, meas_stride,
                    period_sweeps, burn_in_frac=0.2, max_lag=50, signal_timescale_sweeps=None,
                    run_metadata_extra=None):
    """Compute the Section-6 analysis for one case, using a burn-in fraction of the driven
    segment discarded as transient. Reliability tiers (see protocol document Section 6 and
    analyze_response.py's module docstring):
        High confidence: absorbed energy, birth efficiency, net TE(input->birth),
                          tau(input->output) for White/Telegraph.
        Conditional:     TE/correlation involving `drop` (report n_valid alongside).
        Method change:   Sine/Square use phase_lag_at_frequency() instead of a
                          cross-correlation tau, which is not well-defined for a periodic
                          input (confirmed empirically -- see recompute_tau.py and README).

    signal_timescale_sweeps: the input's characteristic time scale in raw sweeps, used for
        the TE reliability check (see analyze_response.assess_te_reliability). Defaults to
        period_sweeps (correct for Sine/Square). For Random telegraph, the caller should pass
        something like 2*dwell_sweeps instead (see dwell_time_sweep.py) -- telegraph has no
        single exact period, but its sign-flip cycle plays the same role.
    """
    if signal_timescale_sweeps is None:
        signal_timescale_sweeps = period_sweeps
    total_driven_sweeps = len(driven_signal)
    te_assessment = assess_te_reliability(signal_timescale_sweeps, meas_stride,
                                            total_driven_sweeps)
    period_to_stride_ratio = te_assessment["samples_per_period"]
    te_reliable = te_assessment["overall"] == "HIGH"

    baseline = {k: rec[k][:driven_start_idx] for k in OUTPUTS}
    driven_full = {k: rec[k][driven_start_idx:] for k in OUTPUTS}
    n_driven_meas = len(driven_full["energy"])
    burn = int(burn_in_frac * n_driven_meas)
    driven = {k: v[burn:] for k, v in driven_full.items()}

    input_meas = delta_K[len(delta_K) - len(driven_signal):][::meas_stride][burn:]
    input_meas = input_meas[:len(driven["energy"])]
    for k in driven:
        driven[k] = driven[k][:len(input_meas)]

    baseline_mean = {k: np.nanmean(v) for k, v in baseline.items()}

    summary = {"case": case}
    summary["input_shannon_entropy_bits"] = signals.shannon_entropy_of_signal(driven_signal)
    summary["baseline_energy"] = baseline_mean["energy"]
    summary["signal_period_to_stride_ratio"] = period_to_stride_ratio
    summary["te_n_cycles"] = te_assessment["n_cycles"]
    summary["te_reliability"] = te_assessment["overall"]  # "HIGH"/"MEDIUM"/"LOW"
    summary["te_reliable"] = te_reliable  # bool 하위호환 (== "HIGH")
    if te_assessment["overall"] != "HIGH":
        print(f"  [{case}] {te_assessment['overall']} TE reliability -- "
              f"samples/period={te_assessment['samples_per_period']:.1f} "
              f"({te_assessment['samples_per_period_tier']}), "
              f"n_cycles={te_assessment['n_cycles']:.1f} "
              f"({te_assessment['n_cycles_tier']}). net_TE is still computed and stored, "
              f"but should be excluded from TE plots/conclusions unless HIGH. "
              f"phase_lag / cross-correlation tau are much less sensitive to this and remain "
              f"usable regardless. Run protocol_check.py before committing to a full sweep "
              f"of this parameter to catch this earlier.")


    for k in OUTPUTS:
        summary[f"baseline_{k}_mean"] = baseline_mean[k]
        summary[f"driven_{k}_mean"] = float(np.nanmean(driven[k]))
        summary[f"delta_{k}"] = summary[f"driven_{k}_mean"] - baseline_mean[k]
        # 채찍GPT 권고: p<0.05뿐 아니라 효과크기(Hedges' g)도 같이 저장 --
        # baseline 구간 원본(burn-in 적용 전, driven_full이 아니라 baseline 그대로) 대
        # driven(burn-in 적용됨) 비교.
        g, n_b, n_d = cohens_d(baseline[k], driven[k])
        summary[f"effect_size_{k}"] = g
        summary[f"effect_size_{k}_label"] = effect_size_label(g)

    d_E = absorbed_energy(driven["energy"], baseline_mean["energy"])
    summary["mean_absorbed_energy"] = float(np.nanmean(d_E))

    # Birth first (high confidence, no NaN ever), then Drop (conditional -- report n_valid).
    for out_name in ["birth", "drop"]:
        eta_t = efficiency(driven[out_name], baseline_mean[out_name],
                            driven["energy"], baseline_mean["energy"])
        summary[f"efficiency_{out_name}_per_absorbedE"] = float(np.nanmean(eta_t))

        this_max_lag = min(max_lag, len(input_meas) // 4)
        if case in ("sine", "square"):
            # Method change for periodic input: phase lag at the known drive frequency,
            # not a cross-correlation argmax (see module docstring for why). Square waves
            # are just as deterministically periodic as sine, so the same reasoning applies
            # -- and this is exactly what's needed to properly handle dwell_sweeps=1 in
            # dwell_time_sweep.py, which is really period_sweeps=2 of this signal family,
            # not a random telegraph process at all.
            omega = omega_from_period(period_sweeps, meas_stride)
            phi, amp_in, amp_out, n_valid = phase_lag_at_frequency(input_meas, driven[out_name],
                                                                     omega)
            summary[f"phase_lag_input_to_{out_name}_rad"] = phi
            summary[f"phase_lag_amp_out_{out_name}"] = amp_out
            summary[f"phase_lag_n_valid_{out_name}"] = n_valid
            summary[f"tau_input_to_{out_name}"] = None  # explicitly not reported for Sine
            summary[f"xcorr_peak_input_to_{out_name}"] = None
            summary[f"xcorr_n_valid_input_to_{out_name}"] = None
        else:
            lags, r, n_valid_arr, tau, r_best, n_valid_best = cross_correlation_lag(
                input_meas, driven[out_name], max_lag=this_max_lag)
            summary[f"tau_input_to_{out_name}"] = tau
            summary[f"xcorr_peak_input_to_{out_name}"] = r_best
            summary[f"xcorr_n_valid_input_to_{out_name}"] = n_valid_best
            summary[f"phase_lag_input_to_{out_name}_rad"] = None
            summary[f"phase_lag_amp_out_{out_name}"] = None
            summary[f"phase_lag_n_valid_{out_name}"] = None

        net_te, te_fwd, te_bwd, n_te_fwd, n_te_bwd = net_transfer_entropy(
            input_meas, driven[out_name])
        summary[f"TE_input_to_{out_name}"] = te_fwd
        summary[f"TE_{out_name}_to_input"] = te_bwd
        summary[f"net_TE_input_{out_name}"] = net_te
        summary[f"TE_n_valid_{out_name}"] = n_te_fwd

    # 채찍GPT 권고: P4/P5로 넘어갈 때 "이 데이터가 정확히 어느 코드/시점으로 만들어졌는지"
    # 추적하기 위한 run metadata. run_metadata_extra로 넘긴 파라미터(seed, period,
    # dwell, L, T, K0, power 등)와 합쳐서 저장한다.
    summary.update(get_run_metadata())
    if run_metadata_extra:
        summary.update(run_metadata_extra)

    return summary


MORPHOLOGY_KEYS = ["info_mean", "info_max", "info_std", "info_skew", "info_kurtosis",
                    "info_corr_length", "info_n_components", "info_largest_component_frac"]


def save_case_timeseries(case, rec, outdir):
    import csv
    path = os.path.join(outdir, f"stage1_{case}_timeseries.csv")
    # Paper 4를 위한 정보-형태(morphology) 컬럼들은 record_morphology=True로 켰을 때만
    # rec에 들어있다 -- 있으면 같이 저장하고, 없으면(기본, 기존 파이프라인과 동일) 건드리지
    # 않는다.
    keys = ["sweep"] + OUTPUTS + [k for k in MORPHOLOGY_KEYS if k in rec]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in zip(*[rec[k] for k in keys]):
            w.writerow(row)
    print(f"  saved {path}")

    if "_field_snapshots" in rec:
        sweeps, fields = rec["_field_snapshots"]
        snap_path = os.path.join(outdir, f"stage1_{case}_field_snapshots.npz")
        np.savez_compressed(snap_path, sweeps=sweeps, fields=fields)
        print(f"  saved {snap_path} ({fields.shape[0]} snapshots, {fields.shape[1]}x"
              f"{fields.shape[2]} each -- for Paper 4 persistence/MI/entropy-production "
              f"후처리용 원본 정보 필드)")


def plot_case(case, rec, driven_start_idx, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(OUTPUTS), 1, figsize=(9, 2.2 * len(OUTPUTS)), sharex=True)
    sweeps = rec["sweep"]
    for ax, key in zip(axes, OUTPUTS):
        ax.plot(sweeps, rec[key], lw=0.8)
        ax.axvline(sweeps[driven_start_idx], color="red", ls="--", lw=1,
                   label="drive starts" if key == OUTPUTS[0] else None)
        ax.set_ylabel(key)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("sweep")
    fig.suptitle(f"Case: {case}")
    fig.tight_layout()
    path = os.path.join(outdir, f"stage1_{case}_timeseries.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def plot_comparison(summaries, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    cases = [s["case"] for s in summaries]
    entropies = [s["input_shannon_entropy_bits"] for s in summaries]
    absorbed = [s["mean_absorbed_energy"] for s in summaries]
    eta_birth = [s["efficiency_birth_per_absorbedE"] for s in summaries]

    axes[0].bar(cases, absorbed, color="tab:blue")
    axes[0].set_title("Mean absorbed energy\n(the key Section-4 comparison)")
    axes[0].set_ylabel("<E_driven> - <E_baseline>")

    axes[1].bar(cases, eta_birth, color="tab:green")
    axes[1].set_title("Efficiency: birth / absorbed energy")

    # Third panel: tau for White/Telegraph (comparable, cross-correlation-based), phase lag
    # for Sine reported separately as text -- these are NOT the same quantity and must not
    # be plotted on one shared numeric axis (see analyze_response.py module docstring).
    non_sine = [s for s in summaries if s["case"] != "sine"]
    sine_summary = next((s for s in summaries if s["case"] == "sine"), None)
    if non_sine:
        ns_entropies = [s["input_shannon_entropy_bits"] for s in non_sine]
        ns_tau = [s["tau_input_to_birth"] for s in non_sine]
        ns_cases = [s["case"] for s in non_sine]
        axes[2].scatter(ns_entropies, ns_tau, s=80)
        for x, y, c in zip(ns_entropies, ns_tau, ns_cases):
            axes[2].annotate(c, (x, y), textcoords="offset points", xytext=(5, 5))
    axes[2].set_xlabel("input Shannon entropy (bits)")
    axes[2].set_ylabel("response time tau (input->birth, measurement steps)")
    title = "Response time vs input entropy\n(White/Telegraph only"
    if sine_summary is not None:
        phi = sine_summary["phase_lag_input_to_birth_rad"]
        title += f"; Sine reported separately: phase lag={phi:+.2f} rad)"
    else:
        title += ")"
    axes[2].set_title(title, fontsize=9)

    fig.tight_layout()
    path = os.path.join(outdir, "stage1_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def check_linearity(L, T, K0, base_power, n_baseline, n_driven, n_therm, meas_stride, seed):
    """Repeat the White-noise case at 0.5x, 1x, 2x power and check that mean absorbed
    |energy| scales roughly proportionally with power (linear-response sanity check)."""
    print("\n=== Linear-response check (White noise, 0.5x / 1x / 2x power) ===")
    results = []
    for mult in [0.5, 1.0, 2.0]:
        p = base_power * mult
        rec, driven_start_idx, driven_signal, delta_K = run_one_case(
            "white", L, T, K0, p, n_baseline, n_driven, n_therm, meas_stride, seed=seed)
        baseline_E = np.nanmean(rec["energy"][:driven_start_idx])
        driven_E = np.nanmean(rec["energy"][driven_start_idx:])
        dE = abs(driven_E - baseline_E)
        results.append((mult, p, dE))
        print(f"  power_mult={mult:.1f}  P={p:.5g}  |Delta E|={dE:.5g}")
    p_ratio = results[2][2] / (results[0][2] + 1e-15)
    print(f"\n  |Delta E| ratio (2.0x power vs 0.5x power, a 4x change in P) = {p_ratio:.2f}x")
    print("  In the linear-response regime, absorbed energy scales with POWER (not amplitude) -- "
          "like Joule heating going as V^2 -- so a 4x increase in P should give roughly a 4x "
          "increase in |Delta E|. A ratio close to 4 is the GOOD sign here (not 2 -- an earlier "
          "version of this message incorrectly said ~2x). Investigate if the ratio is wildly off "
          "(e.g. >10x or <2x), which would indicate you've left the linear-response regime.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=12, help="kagome lattice size (N=3*L^2 sites)")
    ap.add_argument("--T", type=float, default=0.9, help="temperature")
    ap.add_argument("--K0", type=float, default=0.7, help="baseline coupling (Paper 2 value)")
    ap.add_argument("--power", type=float, default=0.01, help="target input power <delta_K^2>")
    ap.add_argument("--n-baseline", type=int, default=2000, help="undriven sweeps")
    ap.add_argument("--n-driven", type=int, default=8000, help="driven sweeps")
    ap.add_argument("--n-therm", type=int, default=2000, help="thermalization sweeps at K0")
    ap.add_argument("--meas-stride", type=int, default=4, help="measure every N sweeps")
    ap.add_argument("--period-sweeps", type=int, default=40, help="sine period, in sweeps")
    ap.add_argument("--dwell-sweeps", type=int, default=20, help="telegraph mean dwell, sweeps")
    ap.add_argument("--burn-in-frac", type=float, default=0.2,
                     help="fraction of the driven segment discarded as transient. Default 0.2 "
                          "is a GUESS -- run pilot_calibration.py first and set this to "
                          "(measured burn-in sweeps) / n_driven instead.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="stage1_results")
    ap.add_argument("--check-linearity", action="store_true",
                     help="run the White-noise power-scaling sanity check and exit")
    ap.add_argument("--record-morphology", action="store_true",
                     help="Paper 4용: 매 측정마다 정보 필드(I(x,y))의 형태(mean/max/std/"
                          "skew/kurtosis/correlation length/connected components)를 같이 "
                          "저장한다. 지금 분석에는 안 쓰지만, 나중에 다시 마이닝 안 해도 "
                          "되게 미리 저장해두는 옵션 (compute_Hmap을 재사용하므로 추가 "
                          "비용은 작음).")
    ap.add_argument("--record-field-snapshots", action="store_true",
                     help="Paper 4용: 정보 필드 I(x,y) 원본을 --snapshot-stride 간격으로 "
                          "통째로 저장한다 (.npz). Persistence/mutual information/entropy "
                          "production처럼 한 시점만으론 계산 못 하는 시공간 분석을 나중에 "
                          "하려면 필요함. 저장 용량이 커지므로 필요할 때만 켜세요.")
    ap.add_argument("--snapshot-stride", type=int, default=None,
                     help="필드 스냅샷 저장 간격(스윕 단위). 기본값은 meas_stride*10.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.check_linearity:
        check_linearity(args.L, args.T, args.K0, args.power, args.n_baseline, args.n_driven,
                         args.n_therm, args.meas_stride, args.seed)
        return

    summaries = []
    for case in CASES:
        rec, driven_start_idx, driven_signal, delta_K = run_one_case(
            case, args.L, args.T, args.K0, args.power, args.n_baseline, args.n_driven,
            args.n_therm, args.meas_stride, seed=args.seed,
            period_sweeps=args.period_sweeps, dwell_sweeps=args.dwell_sweeps,
            record_morphology=args.record_morphology,
            record_field_snapshots=args.record_field_snapshots,
            snapshot_stride=args.snapshot_stride)

        save_case_timeseries(case, rec, args.outdir)
        plot_case(case, rec, driven_start_idx, args.outdir)

        summary = summarize_case(case, rec, driven_start_idx, driven_signal, delta_K,
                                  args.meas_stride, args.period_sweeps,
                                  burn_in_frac=args.burn_in_frac,
                                  signal_timescale_sweeps=signal_timescale_for_case(
                                      case, args.period_sweeps, args.dwell_sweeps))
        summaries.append(summary)

    # write summary CSV
    import csv
    summary_path = os.path.join(args.outdir, "stage1_summary.csv")
    keys = list(summaries[0].keys())
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"\nsaved {summary_path}")

    with open(os.path.join(args.outdir, "stage1_run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    plot_comparison(summaries, args.outdir)

    print("\n=== Headline comparison ===")
    for s in summaries:
        if s["case"] == "sine":
            response_str = f"phase_lag(->birth)={s['phase_lag_input_to_birth_rad']:+.3f} rad"
        else:
            response_str = f"tau(->birth)={s['tau_input_to_birth']}"
        print(f"  {s['case']:10s}  input_H={s['input_shannon_entropy_bits']:.3f} bits  "
              f"absorbed_E={s['mean_absorbed_energy']:+.5g}  "
              f"eta_birth={s['efficiency_birth_per_absorbedE']:+.4g}  "
              f"{response_str}  "
              f"net_TE(->birth)={s['net_TE_input_birth']:+.4f} "
              f"(n={s['TE_n_valid_birth']})")
    print("\nReliability reminder: absorbed energy, birth efficiency, net_TE(->birth), and "
          "White/Telegraph's tau are high-confidence. Sine's response is reported as a phase "
          "lag (not tau -- a cross-correlation argmax is not well-defined for a periodic "
          "input). Anything involving `drop` is conditional on its pairwise-valid sample "
          "count (see stage1_summary.csv's *_n_valid_* columns) since drop is undefined "
          "whenever no vortex is present.")


if __name__ == "__main__":
    main()
