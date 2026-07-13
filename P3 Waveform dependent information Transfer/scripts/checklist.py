"""
Interactive checklist for the Stage-1 workflow -- a menu instead of remembering CLI flags.

Keeps one config (L, T, K0, power, etc.) in memory for the session and lets you tweak just
the one value you care about before re-running a step, rather than retyping every flag.
Every option here just calls the same functions/scripts already in this folder --
nothing new is computed, this is purely a friendlier way to drive them.

Usage:
    python checklist.py
"""

import json
import os
import subprocess
import sys

CONFIG_PATH = "checklist_config.json"

DEFAULT_CONFIG = {
    "L": 48,
    "T": 0.9,
    "K0": 0.7,
    "power": 0.01,
    "n_baseline": 5000,
    "n_driven": 30000,
    "n_therm": 2000,
    "meas_stride": 5,
    "period_sweeps": 40,
    "dwell_sweeps": 20,
    "burn_in_frac": 0.2,
    "seed": 0,
    "outdir": "stage1_results_seed0",
    "record_morphology": 0,       # Paper 4용: 정보 필드 형태(mean/max/std/skew/...) 저장.
                                    # 0=끔, 1=켬 (bool 대신 int로 둔 이유: 이 파일의 config
                                    # 값 편집기가 int/float/str만 다루기 때문)
    "record_field_snapshots": 0,  # Paper 4용: 정보 필드 원본을 .npz로 저장. 0=끔, 1=켬
    "snapshot_stride": 0,         # 0이면 기본값(meas_stride*10) 사용
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        # fill in any keys added since the file was last saved
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def print_config(cfg):
    print("\nCurrent config:")
    for k, v in cfg.items():
        print(f"    {k:14s} = {v}")
    print()


def edit_config(cfg):
    print_config(cfg)
    print("Type 'key value' to change one setting (e.g. 'seed 3'), or press Enter to go back.")
    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            print("  format is: key value")
            continue
        key, val = parts
        if key not in cfg:
            print(f"  unknown key '{key}'. Known keys: {', '.join(cfg.keys())}")
            continue
        old = cfg[key]
        try:
            if isinstance(old, int) and key != "outdir":
                cfg[key] = int(val)
            elif isinstance(old, float):
                cfg[key] = float(val)
            else:
                cfg[key] = val
        except ValueError:
            print(f"  couldn't parse '{val}' as the same type as {old!r}")
            continue
        print(f"  {key}: {old} -> {cfg[key]}")
    save_config(cfg)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    print()
    if ok:
        print("  -> finished with no errors.")
    else:
        print("  -> exited with an error (see output above). Fix before moving on.")
    input("\nPress Enter to return to the menu...")
    return ok


def step_self_tests():
    print("\n=== Step 1: module self-tests (should all pass quickly) ===")
    for mod in ["kagome_lattice.py", "signals.py", "driven_kagome_sim.py",
                "analyze_response.py"]:
        print(f"\n--- {mod} ---")
        run([sys.executable, mod])


def step_linearity(cfg):
    print("\n=== Step 2: linear-response check (White noise, 0.5x/1x/2x power) ===")
    print("Look for the |Delta E| ratio to land close to 4x (not 2x -- see README).")
    cmd = [sys.executable, "run_stage1_experiment.py", "--check-linearity",
           "--L", str(cfg["L"]), "--T", str(cfg["T"]), "--K0", str(cfg["K0"]),
           "--power", str(cfg["power"]), "--n-baseline", str(cfg["n_baseline"]),
           "--n-driven", str(cfg["n_driven"]), "--n-therm", str(cfg["n_therm"]),
           "--meas-stride", str(cfg["meas_stride"]), "--seed", str(cfg["seed"])]
    run(cmd)


def step_main_run(cfg):
    print(f"\n=== Step 3: full Stage-1 run (outdir={cfg['outdir']}) ===")
    cmd = [sys.executable, "run_stage1_experiment.py",
           "--L", str(cfg["L"]), "--T", str(cfg["T"]), "--K0", str(cfg["K0"]),
           "--power", str(cfg["power"]), "--n-baseline", str(cfg["n_baseline"]),
           "--n-driven", str(cfg["n_driven"]), "--n-therm", str(cfg["n_therm"]),
           "--meas-stride", str(cfg["meas_stride"]), "--period-sweeps", str(cfg["period_sweeps"]),
           "--dwell-sweeps", str(cfg["dwell_sweeps"]), "--burn-in-frac", str(cfg["burn_in_frac"]),
           "--seed", str(cfg["seed"]), "--outdir", cfg["outdir"]]
    if cfg.get("record_morphology"):
        cmd.append("--record-morphology")
    if cfg.get("record_field_snapshots"):
        cmd.append("--record-field-snapshots")
    if cfg.get("snapshot_stride"):
        cmd += ["--snapshot-stride", str(cfg["snapshot_stride"])]
    run(cmd)


def step_recompute_tau(cfg):
    print(f"\n=== Step 4: recheck tau / phase lag on saved results in {cfg['outdir']} ===")
    if not os.path.exists(os.path.join(cfg["outdir"], "stage1_run_config.json")):
        print(f"  no results found in {cfg['outdir']} yet -- run Step 3 first.")
        input("\nPress Enter to return to the menu...")
        return
    max_lag = input("  max-lag for the wider cross-correlation window [default 300]: ").strip()
    max_lag = max_lag or "300"
    sweep = input("  optional comma-separated periods to preview for Sine's phase lag "
                  "(Enter to skip): ").strip()
    cmd = [sys.executable, "recompute_tau.py", "--outdir", cfg["outdir"], "--max-lag", max_lag]
    if sweep:
        cmd += ["--sweep-periods", sweep]
    run(cmd)


def step_view_summary(cfg):
    path = os.path.join(cfg["outdir"], "stage1_summary.csv")
    if not os.path.exists(path):
        print(f"\n  no summary found at {path} yet -- run Step 3 first.")
        input("\nPress Enter to return to the menu...")
        return
    print(f"\n=== stage1_summary.csv ({cfg['outdir']}) ===\n")
    try:
        import pandas as pd
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(pd.read_csv(path).T)
    except ImportError:
        with open(path) as f:
            print(f.read())
    input("\nPress Enter to return to the menu...")


def step_new_seed(cfg):
    cur = cfg["seed"]
    nxt = cur + 1
    val = input(f"  next seed [{cur} -> {nxt}]: ").strip()
    cfg["seed"] = int(val) if val else nxt
    cfg["outdir"] = f"stage1_results_seed{cfg['seed']}"
    save_config(cfg)
    print(f"  seed set to {cfg['seed']}, outdir set to {cfg['outdir']}. "
          f"Go to Step 3 to run it.")
    input("\nPress Enter to return to the menu...")


def step_calibration(cfg):
    print("\n=== Step: pilot calibration (equilibration + autocorrelation times) ===")
    print("Run this BEFORE trusting n_therm/meas_stride/burn-in in your config -- it measures "
          "them instead of guessing. Uses your current L/T/K0/power.")
    n_base = input("  sweeps for the baseline scan [default 4000]: ").strip() or "4000"
    n_driv = input("  sweeps for the driven-transient scan [default 4000]: ").strip() or "4000"
    case = input("  drive case to probe (white/sine/telegraph) [default white]: ").strip() \
        or "white"
    outdir = f"pilot_calibration_L{cfg['L']}"
    cmd = [sys.executable, "pilot_calibration.py",
           "--L", str(cfg["L"]), "--T", str(cfg["T"]), "--K0", str(cfg["K0"]),
           "--power", str(cfg["power"]), "--n-sweeps-baseline", n_base,
           "--n-sweeps-driven", n_driv, "--drive-case", case,
           "--period-sweeps", str(cfg["period_sweeps"]), "--dwell-sweeps", str(cfg["dwell_sweeps"]),
           "--seed", str(cfg["seed"]), "--outdir", outdir]
    run(cmd)
    print(f"  Plots and recommendations are in {outdir}/ -- update the config (option 0) "
          f"with n_therm, meas_stride, etc. once you've looked them over.")


def step_multiseed(cfg):
    print("\n=== Step: multi-seed reproducibility check ===")
    print("Runs all 3 cases across several seeds and reports Delta-Birth, absorbed energy, "
          "efficiency, and net TE(->Birth) as mean +/- 95% CI -- this is what actually tells "
          "you whether a pattern from one seed is real or noise.")
    seeds = input("  comma-separated seeds [default 0,1,2,3,4,5,6]: ").strip() \
        or "0,1,2,3,4,5,6"
    outdir = input(f"  outdir [default multiseed_results]: ").strip() or "multiseed_results"
    cmd = [sys.executable, "multiseed_stage1.py", "--seeds", seeds,
           "--L", str(cfg["L"]), "--T", str(cfg["T"]), "--K0", str(cfg["K0"]),
           "--power", str(cfg["power"]), "--n-baseline", str(cfg["n_baseline"]),
           "--n-driven", str(cfg["n_driven"]), "--n-therm", str(cfg["n_therm"]),
           "--meas-stride", str(cfg["meas_stride"]), "--period-sweeps", str(cfg["period_sweeps"]),
           "--dwell-sweeps", str(cfg["dwell_sweeps"]), "--burn-in-frac", str(cfg["burn_in_frac"]),
           "--outdir", outdir]
    run(cmd)
    print(f"  Figures (delta_birth.png, absorbed_energy.png, efficiency.png, "
          f"net_TE_birth.png, all_four_figures.png) and multiseed_summary.csv are in {outdir}/")


def step_dwell_sweep(cfg):
    print("\n=== Step: telegraph dwell-time sweep (entropy 고정, correlation time만 스윕) ===")
    dwells = input("  dwell_sweeps 목록 [default 1,2,5,10,20,40]: ").strip() \
        or "1,2,5,10,20,40"
    seeds = input("  시드 목록 [default 0,1,2,3,4]: ").strip() or "0,1,2,3,4"
    outdir = input("  outdir [default dwell_sweep_results]: ").strip() \
        or "dwell_sweep_results"
    cmd = [sys.executable, "dwell_time_sweep.py", "--dwells", dwells, "--seeds", seeds,
           "--L", str(cfg["L"]), "--T", str(cfg["T"]), "--K0", str(cfg["K0"]),
           "--power", str(cfg["power"]), "--n-baseline", str(cfg["n_baseline"]),
           "--n-driven", str(cfg["n_driven"]), "--n-therm", str(cfg["n_therm"]),
           "--meas-stride", str(cfg["meas_stride"]), "--burn-in-frac", str(cfg["burn_in_frac"]),
           "--outdir", outdir]
    run(cmd)
    print(f"  결과: {outdir}/dwell_sweep_summary.csv, "
          f"dwell_sweep_TE_vs_correlation_time.png, dwell_sweep_TE_vs_tau_signal.png")


def step_square_sweep(cfg):
    print("\n=== Step: square-wave period 스윕 (dwell=1과 sine 사이 빈 구간 채우기) ===")
    periods = input("  period_sweeps 목록 [default 2,4,8,16,32,64,128]: ").strip() \
        or "2,4,8,16,32,64,128"
    seeds = input("  시드 목록 [default 0,1,2,3,4]: ").strip() or "0,1,2,3,4"
    outdir = input("  outdir [default square_sweep_results]: ").strip() \
        or "square_sweep_results"
    cmd = [sys.executable, "square_wave_sweep.py", "--periods", periods, "--seeds", seeds,
           "--L", str(cfg["L"]), "--T", str(cfg["T"]), "--K0", str(cfg["K0"]),
           "--power", str(cfg["power"]), "--n-baseline", str(cfg["n_baseline"]),
           "--n-driven", str(cfg["n_driven"]), "--n-therm", str(cfg["n_therm"]),
           "--meas-stride", str(cfg["meas_stride"]), "--burn-in-frac", str(cfg["burn_in_frac"]),
           "--outdir", outdir]
    run(cmd)
    print(f"  결과: {outdir}/square_sweep_summary.csv, square_sweep_TE_vs_period.png")


def step_protocol_check(cfg):
    print("\n=== Step: 측정 프로토콜 사전 점검 (실험 돌리기 전에 먼저 확인) ===")
    signal_type = input("  신호 종류 (periodic/telegraph) [default periodic]: ").strip() \
        or "periodic"
    cmd = [sys.executable, "protocol_check.py", "--signal-type", signal_type,
           "--meas-stride", str(cfg["meas_stride"])]
    if signal_type == "periodic":
        vals = input("  period_sweeps 목록 [default 2,4,8,16,32,64,128]: ").strip() \
            or "2,4,8,16,32,64,128"
        cmd += ["--timescales", vals]
    else:
        vals = input("  dwell_sweeps 목록 [default 1,2,5,10,20,40]: ").strip() \
            or "1,2,5,10,20,40"
        cmd += ["--dwells", vals]
    run(cmd)


MENU = """
=========================================================
 Paper 3, Stage 1 -- interactive checklist
=========================================================
  0) Show / edit current config
  1) Pilot calibration: measure equilibration + autocorrelation times (do this FIRST)
  2) Module self-tests (run this too, every time you pull new code)
  3) Linear-response check (before trusting any real run)
  4) Run the full Stage-1 experiment (current config, single seed)
  5) Recheck tau / phase lag on saved results (no rerun needed)
  6) View last summary CSV
  7) Bump seed and point outdir at a fresh folder (for repeat single-seed runs)
  8) Multi-seed reproducibility check (Delta-Birth / energy / efficiency / TE, mean +/- CI)
  9) Telegraph dwell-time sweep (entropy 고정, correlation time만 바꿔서 TE 확인)
 10) Square-wave period sweep (dwell=1과 sine 사이의 빈 주파수 구간 채우기)
 11) 측정 프로토콜 사전 점검 (새 신호 추가 전 나이퀴스트 체크 -- 비싼 실험 돌리기 전에!)
  q) Quit
=========================================================
"""


def main():
    cfg = load_config()
    while True:
        print(MENU)
        print_config(cfg)
        choice = input("choice> ").strip().lower()
        if choice == "0":
            edit_config(cfg)
        elif choice == "1":
            step_calibration(cfg)
        elif choice == "2":
            step_self_tests()
        elif choice == "3":
            step_linearity(cfg)
        elif choice == "4":
            step_main_run(cfg)
        elif choice == "5":
            step_recompute_tau(cfg)
        elif choice == "6":
            step_view_summary(cfg)
        elif choice == "7":
            step_new_seed(cfg)
        elif choice == "8":
            step_multiseed(cfg)
        elif choice == "9":
            step_dwell_sweep(cfg)
        elif choice == "10":
            step_square_sweep(cfg)
        elif choice == "11":
            step_protocol_check(cfg)
        elif choice == "q":
            print("bye!")
            break
        else:
            print("didn't recognize that -- pick one of the listed options.")


if __name__ == "__main__":
    main()
