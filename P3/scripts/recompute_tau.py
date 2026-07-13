"""
Recompute cross-correlation response time (tau) / phase lag from ALREADY-SAVED Stage-1
results, with a different max_lag or a direct phase-lag sweep -- no simulation rerun needed.

Why this doesn't need to touch the lattice at all: run_stage1_experiment.py generates each
input signal deterministically from (case, seed, n_driven, power, period_sweeps,
dwell_sweeps) via a freshly-seeded RNG, *before* that same RNG is handed to the (expensive)
MC simulation. So re-seeding an RNG the same way and re-calling the signal generator
reproduces the exact input signal bit-for-bit, with no need to redo any Metropolis sweeps.
The saved stage1_<case>_timeseries.csv already has the output side; this script just
regenerates the input side and re-runs the (cheap) response analysis.

For White/Telegraph: re-runs cross_correlation_lag at a wider --max-lag.
For Sine: does NOT use cross-correlation at all (see analyze_response.py's module
docstring for why tau is not well-defined for a periodic input) -- instead reports the
phase lag at the known drive frequency, and can optionally sweep a few candidate
--period-sweeps values around the recorded one if you want to see how phase lag depends on
drive frequency (a cheap preview of the frequency-response idea in the protocol document's
Section 8 roadmap).

Usage:
    python recompute_tau.py --outdir stage1_results_seed0 --max-lag 300
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import signals
from driven_kagome_sim import build_protocol_schedule
from analyze_response import cross_correlation_lag, phase_lag_at_frequency, omega_from_period


CASES = ["sine", "white", "telegraph"]


def regenerate_signal(case, cfg):
    rng = np.random.default_rng(cfg["seed"])
    if case == "sine":
        return signals.gen_sine(cfg["n_driven"], cfg["power"],
                                 period_sweeps=cfg["period_sweeps"], rng=rng)
    elif case == "white":
        return signals.gen_white_noise(cfg["n_driven"], cfg["power"], rng=rng)
    elif case == "telegraph":
        return signals.gen_random_telegraph(cfg["n_driven"], cfg["power"],
                                             dwell_sweeps=cfg["dwell_sweeps"], rng=rng)
    else:
        raise ValueError(case)


def align_input_output(case, cfg, outdir, burn_in_frac):
    df = pd.read_csv(os.path.join(outdir, f"stage1_{case}_timeseries.csv"))
    driven_signal = regenerate_signal(case, cfg)
    signals.check_power(driven_signal, cfg["power"], label=f"{case} (regenerated)")

    delta_K, driven_start_sweep = build_protocol_schedule(
        cfg["n_baseline"], cfg["n_driven"], driven_signal)

    sweeps = df["sweep"].to_numpy()
    driven_start_idx = int(np.searchsorted(sweeps, driven_start_sweep))
    n_driven_meas = len(sweeps) - driven_start_idx
    burn = int(burn_in_frac * n_driven_meas)

    outputs = ["energy", "drop", "birth", "info", "helicity"]
    driven = {k: df[k].to_numpy()[driven_start_idx:][burn:] for k in outputs}

    input_meas = delta_K[len(delta_K) - len(driven_signal):][::cfg["meas_stride"]][burn:]
    n = min(len(input_meas), len(driven["energy"]))
    input_meas = input_meas[:n]
    for k in driven:
        driven[k] = driven[k][:n]
    return input_meas, driven


def recheck_xcorr_case(case, cfg, outdir, max_lag, burn_in_frac):
    input_meas, driven = align_input_output(case, cfg, outdir, burn_in_frac)
    results = {}
    n = len(input_meas)
    this_max_lag = min(max_lag, n // 3)
    if this_max_lag < max_lag:
        print(f"  [{case}] note: requested max_lag={max_lag} truncated to {this_max_lag} "
              f"(only {n} usable samples after burn-in)")
    for out_name in ["birth", "drop", "energy"]:
        lags, r, n_valid_arr, tau, r_best, n_valid_best = cross_correlation_lag(
            input_meas, driven[out_name], max_lag=this_max_lag)
        results[out_name] = (lags, r, tau, r_best, n_valid_best)
    return results


def plot_xcorr(case, results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (out_name, (lags, r, tau, r_best, n_valid)) in zip(axes, results.items()):
        ax.plot(lags, r, lw=1)
        if tau is not None:
            ax.axvline(tau, color="red", ls="--", lw=1, label=f"peak tau={tau} (n={n_valid})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("lag (measurement steps)")
        ax.set_ylabel("cross-correlation")
        ax.set_title(f"{case}: input -> {out_name}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"recheck_xcorr_{case}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def recheck_phase_sine(cfg, outdir, burn_in_frac, period_sweeps_list=None):
    input_meas, driven = align_input_output("sine", cfg, outdir, burn_in_frac)
    periods = period_sweeps_list or [cfg["period_sweeps"]]
    print("\n=== Sine: phase lag phi(omega) [reported instead of tau -- see module docstring] ===")
    rows = []
    for period in periods:
        omega = omega_from_period(period, cfg["meas_stride"])
        for out_name in ["birth", "drop", "energy"]:
            phi, amp_in, amp_out, n_valid = phase_lag_at_frequency(
                input_meas, driven[out_name], omega)
            print(f"  period_sweeps={period:6.1f}  input->{out_name:8s}  "
                  f"phi={phi:+.3f} rad  amp_out={amp_out:.4g}  n_valid={n_valid}")
            rows.append({"period_sweeps": period, "output": out_name, "phi_rad": phi,
                         "amp_out": amp_out, "n_valid": n_valid})
    if len(periods) > 1:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "recheck_phase_sweep_sine.csv"),
                                   index=False)
        print(f"  saved {os.path.join(outdir, 'recheck_phase_sweep_sine.csv')}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True,
                     help="directory containing stage1_run_config.json and the per-case CSVs")
    ap.add_argument("--max-lag", type=int, default=300,
                     help="lag range for White/Telegraph cross-correlation (measurement steps)")
    ap.add_argument("--burn-in-frac", type=float, default=0.2,
                     help="must match the value used in the original run (default 0.2)")
    ap.add_argument("--sweep-periods", type=str, default=None,
                     help="comma-separated list of period_sweeps values to additionally probe "
                          "for the Sine phase-lag preview, e.g. '20,30,40,60,80' (does NOT "
                          "rerun the simulation -- the actual driven run only ever used the "
                          "recorded period_sweeps; this reprojects the SAME saved input/output "
                          "onto other candidate frequencies, which is only meaningful near the "
                          "recorded period -- a real frequency scan needs separate runs at each "
                          "period, this is just a cheap sanity check of the method).")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "stage1_run_config.json")) as f:
        cfg = json.load(f)

    print(f"Loaded config: seed={cfg['seed']}, power={cfg['power']}, "
          f"period_sweeps={cfg['period_sweeps']}, dwell_sweeps={cfg['dwell_sweeps']}, "
          f"meas_stride={cfg['meas_stride']}\n")

    print("=== White / Telegraph: recomputed tau, wider lag window ===")
    for case in ["white", "telegraph"]:
        results = recheck_xcorr_case(case, cfg, args.outdir, args.max_lag, args.burn_in_frac)
        for out_name, (lags, r, tau, r_best, n_valid) in results.items():
            print(f"  {case:10s} input->{out_name:8s}  tau={tau}  peak_r={r_best:+.3f}  "
                  f"n_valid={n_valid}")
        plot_xcorr(case, results, args.outdir)
        print()

    periods = None
    if args.sweep_periods:
        periods = [float(p) for p in args.sweep_periods.split(",")]
    recheck_phase_sine(cfg, args.outdir, args.burn_in_frac, periods)

    print("\nFor White/Telegraph: a genuine response time shows a single clear peak away from "
          "lag=0 that decays on both sides in recheck_xcorr_*.png, AND tau should not change "
          "when you increase --max-lag further (if it does, that case has more structure than "
          "expected and should be looked at more closely -- White/Telegraph did not show this "
          "instability in the pilot run). Sine intentionally uses phase lag, not tau.")


if __name__ == "__main__":
    main()
