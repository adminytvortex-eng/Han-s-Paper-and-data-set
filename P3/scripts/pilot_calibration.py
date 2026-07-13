"""
Pilot calibration for Paper 3, Stage 1 -- run this BEFORE any production run.

Answers four questions a first-time experimentalist should always ask before trusting a
Monte Carlo measurement (this is exactly the logic behind Paper 1's Section 2.4/2.5 dynamic
fine-time scan and N_THERM verification, extended here to also cover the driven segment,
which Paper 1 never needed):

  1. Equilibration time (N_THERM): starting from a random configuration at K0 (no drive),
     how many sweeps until energy/birth stop drifting and start merely fluctuating?
  2. Baseline autocorrelation time (tau_auto): once equilibrated, how many sweeps apart do
     two measurements need to be to count as roughly independent samples?
  3. Driven-state transient time: after switching the drive on, how many sweeps until the
     system settles into its NEW (driven) steady state? This sets burn_in_frac -- currently
     a guessed 20% in run_stage1_experiment.py -- to a measured value instead.
  4. Driven-state autocorrelation time: same question as (2), but during driving (the drive
     itself can change how correlated consecutive configurations are).

Together these turn n_therm, meas_stride, burn_in_frac, and n_driven from guesses into
measured, defensible choices -- and tell you directly how many INDEPENDENT samples you
actually have to work with for a given n_driven (which matters a lot for the discrete-bin
transfer-entropy estimator: too few independent samples and TE estimates are unreliable
regardless of how good the underlying effect is).

Usage:
    python pilot_calibration.py --L 48 --T 0.9 --K0 0.7 --power 0.01
"""

import argparse
import os

import numpy as np

from kagome_lattice import KagomeLattice
from driven_kagome_sim import (metropolis_sweep, mean_energy, plaquette_winding_rounded,
                                vortex_plaquette_ids)
import signals


# ------------------------------------------------------------------------------
# Autocorrelation time (Sokal automatic windowing -- standard, well-justified method)
# ------------------------------------------------------------------------------

def autocorrelation_function(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.mean(x ** 2)
    if var < 1e-15:
        return np.ones(max_lag + 1)  # constant series -> undefined, treat as fully correlated
    n = len(x)
    C = np.empty(max_lag + 1)
    C[0] = 1.0
    for lag in range(1, max_lag + 1):
        C[lag] = np.mean(x[:n - lag] * x[lag:]) / var
    return C


def integrated_autocorr_time(C, c=5):
    """Sokal's automatic windowing: tau_int = 0.5 + sum(C[1..M]), stopping at the first M
    with M >= c*tau_int(M). Returns (tau_int, window_used, converged: bool).
    """
    tau = 0.5
    for M in range(1, len(C)):
        tau += C[M]
        if M >= c * tau:
            return float(tau), M, True
    return float(tau), len(C) - 1, False  # did not converge within max_lag -- need more lags


def measure_autocorr_time(series, max_lag, label=""):
    n = len(series)
    safe_max_lag = min(max_lag, max(1, n // 4))
    if safe_max_lag < max_lag:
        print(f"  [{label}] note: only {n} post-equilibration samples available -- "
              f"max_lag capped from {max_lag} to {safe_max_lag} (need a longer run, or a "
              f"shorter detected equilibration point, for the full requested lag range).")
    C = autocorrelation_function(series, safe_max_lag)
    tau, window, converged = integrated_autocorr_time(C)
    if not converged:
        print(f"  [{label}] WARNING: autocorrelation did not converge within "
              f"max_lag={safe_max_lag} -- tau_int={tau:.2f} is a LOWER BOUND only. "
              f"Re-run with a larger --max-lag-acf and/or --n-sweeps-baseline.")
    else:
        print(f"  [{label}] tau_int = {tau:.2f} sweeps (Sokal window M={window}, "
              f"n_samples={n})")
    return tau, C


# ------------------------------------------------------------------------------
# Step 1+2: baseline equilibration and autocorrelation
# ------------------------------------------------------------------------------

def scan_baseline(L, T, K0, n_sweeps, rng, record_every=1):
    lat = KagomeLattice(L)
    phi = rng.uniform(-np.pi, np.pi, size=lat.N)
    energy = np.empty(n_sweeps // record_every)
    birth_frac = np.empty(n_sweeps // record_every)  # instantaneous vortex fraction, not
                                                        # the birth *rate* -- cheap per-sweep
                                                        # proxy good enough for calibration
    idx = 0
    for t in range(n_sweeps):
        phi = metropolis_sweep(phi, lat.neighbors, K0, T, rng)
        if t % record_every == 0:
            energy[idx] = mean_energy(phi, lat.bonds, K0)
            w = plaquette_winding_rounded(phi, lat.plaquettes)
            birth_frac[idx] = np.count_nonzero(w) / lat.n_plaq
            idx += 1
    return energy[:idx], birth_frac[:idx]


def detect_equilibration(series, tau_auto_est, ref_frac=0.25, n_check=20, tol_sigma=2.5):
    """Heuristic equilibration-point detector: take the mean of the LAST ref_frac of the
    series as the 'converged' reference, then for each of n_check candidate cut points i,
    test whether the AGGREGATE mean of series[cut_i:] is statistically consistent with the
    reference mean (a single two-sample comparison per candidate, with standard errors
    corrected for autocorrelation via tau_auto_est -- effective N ~ length/(2*tau_auto)).
    Report the earliest cut point that passes.

    Deliberately NOT implemented as "every individual block from i onward must
    independently pass its own test": with ~20 near-independent per-block comparisons even
    at a 2.5-sigma threshold, the chance that at least one block anywhere late in the series
    fails by pure noise is large (a multiple-comparisons problem), and a single unlucky block
    poisons every candidate cut point before it. This was caught directly during development:
    a driven series whose block means visibly had no trend at all (confirmed by eye) still
    got flagged as "not equilibrated until 75% through the run" under the per-block version,
    traced to exactly one noisy block failing its individual 2.5-sigma test. The aggregate
    version tests the thing we actually care about (is the remaining data consistent with the
    reference, taken as a whole) instead of a strictly harder, noisier compound question.

    Always look at the printed plot too -- this is a diagnostic aid, not a replacement for
    looking at the data (protocol document Section 5).
    """
    n = len(series)
    n_eff_factor = max(1.0, 2.0 * tau_auto_est)

    ref_start = int(n * (1 - ref_frac))
    ref_seg = series[ref_start:]
    ref_mean = ref_seg.mean()
    ref_n_eff = max(1.0, len(ref_seg) / n_eff_factor)
    ref_sem = ref_seg.std() / np.sqrt(ref_n_eff)

    cut_points = np.linspace(0, ref_start, n_check, endpoint=False).astype(int)
    for cut in cut_points:
        seg = series[cut:]
        n_eff = max(1.0, len(seg) / n_eff_factor)
        seg_mean = seg.mean()
        seg_sem = seg.std() / np.sqrt(n_eff)
        combined_sem = np.sqrt(ref_sem ** 2 + seg_sem ** 2)
        if abs(seg_mean - ref_mean) < tol_sigma * max(combined_sem, 1e-12):
            return int(cut)
    return None  # never stabilized by this criterion -- run longer or inspect the plot


def quick_tau_estimate(series, max_lag=50):
    """Cheap, rough autocorrelation-time estimate (whole series, no equilibration-aware
    windowing) used only to correct standard errors in detect_equilibration -- NOT the final
    reported tau_auto (that comes from measure_autocorr_time on the post-equilibration tail).
    """
    safe_max_lag = min(max_lag, max(1, len(series) // 4))
    C = autocorrelation_function(series, safe_max_lag)
    tau, _, _ = integrated_autocorr_time(C)
    return max(tau, 0.5)


def step_baseline(L, T, K0, n_sweeps, rng, max_lag_acf):
    print(f"\n=== Step 1+2: baseline equilibration + autocorrelation (L={L}, T={T}, K0={K0}) ===")
    print(f"Running {n_sweeps} sweeps from a random configuration...")
    energy, birth_frac = scan_baseline(L, T, K0, n_sweeps, rng)

    tau_est_energy = quick_tau_estimate(energy)
    tau_est_birth = quick_tau_estimate(birth_frac)
    eq_energy = detect_equilibration(energy, tau_est_energy)
    eq_birth = detect_equilibration(birth_frac, tau_est_birth)
    print(f"  detected equilibration sweep (energy): {eq_energy}")
    print(f"  detected equilibration sweep (birth fraction): {eq_birth}")
    candidates = [v for v in [eq_energy, eq_birth] if v is not None]
    n_therm_candidate = max(candidates) if candidates else None
    if n_therm_candidate is None:
        print("  WARNING: neither series looked equilibrated by the automatic heuristic -- "
              "inspect baseline_equilibration.png by eye, and/or increase --n-sweeps-baseline.")
    else:
        print(f"  -> candidate N_THERM (max of the two, no safety margin yet): "
              f"{n_therm_candidate} sweeps")

    # autocorrelation on the back half only (post-equilibration, if we found one; otherwise
    # just use the back half of the whole run as a best effort)
    tail_start = n_therm_candidate if n_therm_candidate is not None else n_sweeps // 2
    tau_energy, C_energy = measure_autocorr_time(energy[tail_start:], max_lag_acf,
                                                   label="baseline energy")
    tau_birth, C_birth = measure_autocorr_time(birth_frac[tail_start:], max_lag_acf,
                                                 label="baseline birth-fraction")
    tau_auto_baseline = max(tau_energy, tau_birth)

    return {
        "energy": energy, "birth_frac": birth_frac,
        "eq_energy": eq_energy, "eq_birth": eq_birth,
        "n_therm_candidate": n_therm_candidate,
        "tau_energy": tau_energy, "tau_birth": tau_birth,
        "tau_auto_baseline": tau_auto_baseline,
        "C_energy": C_energy, "C_birth": C_birth,
    }


# ------------------------------------------------------------------------------
# Step 3+4: driven transient and driven-state autocorrelation
# ------------------------------------------------------------------------------

def scan_driven(L, T, K0, power, n_therm, n_pre, n_post, rng, record_every=1,
                 case="white", period_sweeps=40, dwell_sweeps=20):
    lat = KagomeLattice(L)
    phi = rng.uniform(-np.pi, np.pi, size=lat.N)
    for _ in range(n_therm):
        phi = metropolis_sweep(phi, lat.neighbors, K0, T, rng)

    if case == "white":
        driven_signal = signals.gen_white_noise(n_post, power, rng)
    elif case == "sine":
        driven_signal = signals.gen_sine(n_post, power, period_sweeps=period_sweeps, rng=rng)
    elif case == "telegraph":
        driven_signal = signals.gen_random_telegraph(n_post, power, dwell_sweeps, rng)
    else:
        raise ValueError(case)
    delta_K = np.concatenate([np.zeros(n_pre), driven_signal])
    n_total = len(delta_K)

    energy = np.empty(n_total // record_every)
    birth_frac = np.empty(n_total // record_every)
    idx = 0
    for t in range(n_total):
        K_t = K0 + delta_K[t]
        phi = metropolis_sweep(phi, lat.neighbors, K_t, T, rng)
        if t % record_every == 0:
            energy[idx] = mean_energy(phi, lat.bonds, K0)
            w = plaquette_winding_rounded(phi, lat.plaquettes)
            birth_frac[idx] = np.count_nonzero(w) / lat.n_plaq
            idx += 1
    return energy[:idx], birth_frac[:idx], n_pre


def step_driven(L, T, K0, power, n_therm, n_pre, n_post, rng, max_lag_acf, case="white",
                 period_sweeps=40, dwell_sweeps=20):
    print(f"\n=== Step 3+4: driven transient + driven-state autocorrelation "
          f"(case={case}, power={power}) ===")
    print(f"Equilibrating ({n_therm} sweeps), then {n_pre} undriven + {n_post} driven sweeps...")
    energy, birth_frac, drive_start = scan_driven(L, T, K0, power, n_therm, n_pre, n_post, rng,
                                                    case=case, period_sweeps=period_sweeps,
                                                    dwell_sweeps=dwell_sweeps)

    pre_energy_mean = energy[:drive_start].mean()
    pre_birth_mean = birth_frac[:drive_start].mean()
    print(f"  pre-drive baseline: <energy>={pre_energy_mean:.5g}, <birth_frac>={pre_birth_mean:.5g}")

    driven_energy = energy[drive_start:]
    driven_birth = birth_frac[drive_start:]
    tau_est_energy_d = quick_tau_estimate(driven_energy)
    tau_est_birth_d = quick_tau_estimate(driven_birth)
    transient_energy = detect_equilibration(driven_energy, tau_est_energy_d)
    transient_birth = detect_equilibration(driven_birth, tau_est_birth_d)
    print(f"  detected driven transient length (energy): {transient_energy}")
    print(f"  detected driven transient length (birth fraction): {transient_birth}")
    candidates = [v for v in [transient_energy, transient_birth] if v is not None]
    transient_candidate = max(candidates) if candidates else None
    if transient_candidate is None:
        print("  WARNING: driven series didn't look settled by the automatic heuristic -- "
              "inspect driven_transient.png, and/or increase --n-sweeps-driven.")
    else:
        print(f"  -> candidate burn-in length (sweeps, no safety margin yet): "
              f"{transient_candidate}")

    tail_start = transient_candidate if transient_candidate is not None else len(driven_energy)//2
    tau_energy_d, C_energy_d = measure_autocorr_time(driven_energy[tail_start:], max_lag_acf,
                                                        label="driven energy")
    tau_birth_d, C_birth_d = measure_autocorr_time(driven_birth[tail_start:], max_lag_acf,
                                                     label="driven birth-fraction")
    tau_auto_driven = max(tau_energy_d, tau_birth_d)

    return {
        "energy": energy, "birth_frac": birth_frac, "drive_start": drive_start,
        "transient_candidate": transient_candidate,
        "tau_energy_driven": tau_energy_d, "tau_birth_driven": tau_birth_d,
        "tau_auto_driven": tau_auto_driven,
    }


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

def make_plots(baseline_result, driven_result, outdir, safety_n_therm, period_sweeps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    axes[0].plot(baseline_result["energy"], lw=0.7)
    axes[1].plot(baseline_result["birth_frac"], lw=0.7)
    for ax, key, label in [(axes[0], "eq_energy", "energy"), (axes[1], "eq_birth", "birth_frac")]:
        if baseline_result[key] is not None:
            ax.axvline(baseline_result[key], color="orange", ls="--",
                       label=f"detected equilibration ({baseline_result[key]})")
        ax.axvline(safety_n_therm, color="red", ls="-",
                   label=f"recommended N_THERM w/ safety margin ({safety_n_therm})")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("sweep")
    fig.suptitle("Baseline equilibration (Step 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "baseline_equilibration.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    max_lag = len(baseline_result["C_energy"]) - 1
    ax.plot(baseline_result["C_energy"], label=f"energy (tau={baseline_result['tau_energy']:.1f})")
    ax.plot(baseline_result["C_birth"], label=f"birth_frac (tau={baseline_result['tau_birth']:.1f})")
    ax.axhline(1 / np.e, color="gray", ls=":", lw=1, label="1/e")
    ax.set_xlabel("lag (sweeps)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Baseline autocorrelation function (Step 2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "baseline_autocorrelation.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    ds = driven_result["drive_start"]
    axes[0].plot(driven_result["energy"], lw=0.7)
    axes[1].plot(driven_result["birth_frac"], lw=0.7)
    for ax in axes:
        ax.axvline(ds, color="red", ls="--", label="drive turns on")
        if driven_result["transient_candidate"] is not None:
            ax.axvline(ds + driven_result["transient_candidate"], color="orange", ls="--",
                       label=f"detected new steady state (+{driven_result['transient_candidate']})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("energy")
    axes[1].set_ylabel("birth_frac")
    axes[-1].set_xlabel("sweep")
    fig.suptitle("Driven transient (Step 3)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "driven_transient.png"), dpi=130)
    plt.close(fig)

    print(f"\nPlots saved to {outdir}/: baseline_equilibration.png, "
          f"baseline_autocorrelation.png, driven_transient.png")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--T", type=float, default=0.9)
    ap.add_argument("--K0", type=float, default=0.7)
    ap.add_argument("--power", type=float, default=0.01)
    ap.add_argument("--n-sweeps-baseline", type=int, default=3000,
                     help="total sweeps for the equilibration+autocorrelation scan")
    ap.add_argument("--n-sweeps-driven", type=int, default=3000,
                     help="driven sweeps for the transient+autocorrelation scan")
    ap.add_argument("--max-lag-acf", type=int, default=200,
                     help="max lag searched for autocorrelation time -- increase if you see "
                          "a 'did not converge' warning")
    ap.add_argument("--drive-case", choices=["white", "sine", "telegraph"], default="white",
                     help="which input case to use for the driven-transient scan (white is a "
                          "reasonable default probe; re-run with sine/telegraph if you want "
                          "case-specific transient/autocorrelation numbers)")
    ap.add_argument("--period-sweeps", type=int, default=40)
    ap.add_argument("--dwell-sweeps", type=int, default=20)
    ap.add_argument("--safety-factor", type=float, default=10.0,
                     help="N_THERM and burn-in are reported as (detected value) x this "
                          "factor, matching Paper 1's own N_THERM/tau_auto safety margin "
                          "(6-40x) -- see the protocol document.")
    ap.add_argument("--target-independent-samples", type=int, default=2000,
                     help="how many independent driven-state samples you want after burn-in, "
                          "for a reasonably powered transfer-entropy estimate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="pilot_calibration")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    baseline = step_baseline(args.L, args.T, args.K0, args.n_sweeps_baseline, rng,
                              args.max_lag_acf)

    n_therm_base = (baseline["n_therm_candidate"] if baseline["n_therm_candidate"] is not None
                    else args.n_sweeps_baseline // 2)
    n_therm_recommended = int(np.ceil(n_therm_base))
    n_therm_recommended = max(n_therm_recommended,
                               int(np.ceil(args.safety_factor * baseline["tau_auto_baseline"])))

    driven = step_driven(args.L, args.T, args.K0, args.power, n_therm_recommended,
                          n_pre=200, n_post=args.n_sweeps_driven, rng=rng,
                          max_lag_acf=args.max_lag_acf, case=args.drive_case,
                          period_sweeps=args.period_sweeps, dwell_sweeps=args.dwell_sweeps)

    transient_base = (driven["transient_candidate"] if driven["transient_candidate"] is not None
                      else args.n_sweeps_driven // 2)
    burn_in_recommended = int(np.ceil(args.safety_factor * 0.3 * transient_base))
    burn_in_floor = int(np.ceil(5 * driven["tau_auto_driven"]))
    if burn_in_recommended < burn_in_floor:
        print(f"\n  (raising burn-in from {burn_in_recommended} to a floor of "
              f"{burn_in_floor} = 5x tau_auto_driven, since a detected transient of "
              f"~{transient_base} sweeps is close to the autocorrelation time itself and "
              f"shouldn't be trusted down to 0)")
        burn_in_recommended = burn_in_floor
    # (0.3x the usual safety factor for burn-in -- unlike N_THERM, which only needs to run
    # once, burn-in sweeps are paid for on every seed/case, so an overly generous margin here
    # is much more expensive in aggregate; use the plot to sanity-check this is still enough)

    make_plots(baseline, driven, args.outdir, n_therm_recommended, args.period_sweeps)

    meas_stride_for_independence = max(1, int(np.ceil(driven["tau_auto_driven"])))
    meas_stride_for_sine_resolution = max(1, args.period_sweeps // 8)  # >=8 samples/period

    n_independent_available = lambda n_driven, stride: max(
        0, (n_driven - burn_in_recommended)) // max(stride, driven["tau_auto_driven"])
    n_driven_for_target = int(np.ceil(
        burn_in_recommended
        + args.target_independent_samples * max(meas_stride_for_independence,
                                                  driven["tau_auto_driven"])))

    print("\n" + "=" * 70)
    print("RECOMMENDED SETTINGS (measured, not guessed -- update your config with these)")
    print("=" * 70)
    print(f"  n_therm        = {n_therm_recommended}   "
          f"(detected equilibration x safety, or {args.safety_factor}x tau_auto_baseline="
          f"{baseline['tau_auto_baseline']:.1f}, whichever larger)")
    print(f"  burn_in sweeps = {burn_in_recommended}   "
          f"(detected driven transient x reduced safety factor -- see code comment for why "
          f"this margin is smaller than N_THERM's)")
    print(f"  tau_auto (baseline) = {baseline['tau_auto_baseline']:.1f} sweeps")
    print(f"  tau_auto (driven, {args.drive_case}) = {driven['tau_auto_driven']:.1f} sweeps")
    print(f"\n  meas_stride: there are TWO competing requirements --")
    print(f"    - for independent samples (TE, efficiency): stride >= tau_auto_driven "
          f"-> meas_stride >= {meas_stride_for_independence}")
    print(f"    - for resolving the Sine period ({args.period_sweeps} sweeps) with >=8 "
          f"samples/cycle -> meas_stride <= {meas_stride_for_sine_resolution}")
    if meas_stride_for_independence > meas_stride_for_sine_resolution:
        print(f"    -> THESE CONFLICT ({meas_stride_for_independence} > "
              f"{meas_stride_for_sine_resolution}). Consider: (a) a coarser meas_stride for "
              f"Birth/TE-focused analysis and a separate finer-strided re-run just for the "
              f"Sine phase-lag measurement, or (b) increasing the Sine period so 8 samples/"
              f"cycle stays compatible with the independence stride.")
    else:
        print(f"    -> compatible: pick a meas_stride between {meas_stride_for_independence} "
              f"and {meas_stride_for_sine_resolution}.")
    print(f"\n  n_driven >= {n_driven_for_target} to get ~{args.target_independent_samples} "
          f"independent post-burn-in samples (at stride=tau_auto_driven)")
    example_burn_frac = burn_in_recommended / n_driven_for_target
    print(f"\n  --burn-in-frac to pass to run_stage1_experiment.py: with n_driven="
          f"{n_driven_for_target}, use --burn-in-frac {example_burn_frac:.3f} "
          f"(= {burn_in_recommended} burn-in sweeps / {n_driven_for_target} total driven "
          f"sweeps). Recompute this ratio if you end up using a different --n-driven than "
          f"the one suggested above -- burn_in_frac scales with it, the raw sweep count "
          f"({burn_in_recommended}) is the one that should stay fixed.")
    print("\nAs always: LOOK AT THE PLOTS before trusting these numbers (protocol document "
          "Section 5) -- the automatic equilibration/transient detector is a starting point, "
          "not a substitute for your own judgement.")


if __name__ == "__main__":
    main()
