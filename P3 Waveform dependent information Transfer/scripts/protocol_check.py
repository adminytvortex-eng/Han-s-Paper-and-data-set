"""
측정 프로토콜 검증 체크리스트 v2 (Measurement Protocol Validation Checklist).

채찍GPT 피드백을 반영해서 v1(단순 period/stride>=8 이분법)에서 3개 지표 x 3단계로
업그레이드했다:

  1. Samples per period  = timescale / meas_stride        -- 한 주기를 얼마나 세밀하게 쟀나
  2. Number of cycles     = total_sweeps / timescale       -- 그 주기가 몇 번 반복됐나
  3. tau_signal / meas_stride (telegraph류만)              -- 실측 상관시간이 stride보다 느린가

각 지표는 FAIL/WARNING/PASS 3단계로 판정되고, 종합 등급(Expected TE reliability)은
셋 중 가장 나쁜 것을 따른다 (하나라도 FAIL -> LOW, WARNING만 있으면 -> MEDIUM,
전부 PASS -> HIGH).

비싼 시뮬레이션을 실제로 돌리기 *전에* 이 스크립트로 먼저 파라미터 조합을 걸러낸다.

사용법:
    # 사인/사각파처럼 뚜렷한 주기가 있는 입력 (total_sweeps는 n_driven과 맞춰서)
    python protocol_check.py --signal-type periodic \\
        --timescales 2,4,8,16,32,64,96,128,192,256,512,2048 \\
        --meas-stride 5 --total-sweeps 35000

    # random telegraph (timescale = 2*dwell_sweeps, tau_signal도 실제로 재서 같이 판정)
    python protocol_check.py --signal-type telegraph --dwells 1,2,5,10,20,40 \\
        --meas-stride 5 --total-sweeps 35000 --power 0.01

    # 주어진 total_sweeps/meas_stride에 맞는 "안전한" period들을 로그 스케일로 자동 추천
    python protocol_check.py --auto-suggest --meas-stride 5 --total-sweeps 35000
"""

import argparse

import numpy as np

from analyze_response import assess_te_reliability


TIER_MARK = {"PASS": "✅ PASS", "WARNING": "⚠️  WARNING", "FAIL": "❌ FAIL"}
OVERALL_MARK = {"HIGH": "HIGH (안심하고 TE 사용)", "MEDIUM": "MEDIUM (조건부 -- 결과에 명시 필요)",
                "LOW": "LOW (TE 분석에서 제외 권장)"}


def measure_tau_for_telegraph(dwell_sweeps, power, meas_stride, seed=0, n_sweeps=20000):
    """dwell_time_sweep.py의 measure_signal_tau를 재사용해서, 실제로 telegraph 신호를
    생성하고 그 실측 자기상관시간을 잰다 (이론적 근사인 2*dwell이 아니라 실제 값).
    """
    import signals
    from dwell_time_sweep import measure_signal_tau

    rng = np.random.default_rng(seed)
    sig = signals.gen_random_telegraph(n_sweeps, power, dwell_sweeps, rng)
    return measure_signal_tau(sig)


def print_one(label, r, tau_note=""):
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")
    print(f"  ① Samples per period  : {r['samples_per_period']:8.1f}   "
          f"{TIER_MARK[r['samples_per_period_tier']]}   (기준: FAIL<5, WARNING<10, PASS>=10)")
    print(f"  ② Number of cycles    : {r['n_cycles']:8.1f}   "
          f"{TIER_MARK[r['n_cycles_tier']]}   (기준: FAIL<20, WARNING<50, PASS>=50)")
    if r["tau_tier"] is not None:
        print(f"  ③ tau_signal/stride   : {r['tau_ratio']:8.2f}   "
              f"{TIER_MARK[r['tau_tier']]}   (기준: WARNING if <1, else PASS){tau_note}")
    print(f"  -> Expected TE reliability: {OVERALL_MARK[r['overall']]}")


def run_periodic(timescales, meas_stride, total_sweeps):
    results = []
    for t in timescales:
        r = assess_te_reliability(t, meas_stride, total_sweeps)
        print_one(f"period_sweeps={t:.0f}", r)
        results.append((t, r))
    return results


def run_telegraph(dwells, meas_stride, total_sweeps, power, seed):
    results = []
    for d in dwells:
        timescale = 2 * d
        tau = measure_tau_for_telegraph(d, power, meas_stride, seed=seed)
        r = assess_te_reliability(timescale, meas_stride, total_sweeps, tau_signal=tau)
        print_one(f"dwell_sweeps={d:.0f}  (timescale=2*dwell={timescale:.0f}, "
                  f"실측 tau_signal={tau:.2f})", r)
        results.append((d, r))
    return results


def auto_suggest(meas_stride, total_sweeps, n_suggest=6):
    """total_sweeps/meas_stride에 맞춰 overall=HIGH 인 period들을 로그 스케일에 가깝게
    자동으로 추천한다 (2의 거듭제곱 후보들을 먼저 보고, 그중 PASS인 것들을 고른다).
    """
    print(f"\n{'='*64}")
    print(f"  자동 추천 (meas_stride={meas_stride}, total_sweeps={total_sweeps})")
    print(f"{'='*64}")
    candidates = [2 ** k for k in range(1, 16)]  # 2,4,8,...,32768
    good = []
    for c in candidates:
        r = assess_te_reliability(c, meas_stride, total_sweeps)
        if r["overall"] == "HIGH":
            good.append(c)
    if not good:
        print("  HIGH 등급을 만족하는 2의 거듭제곱 period가 없습니다 -- meas_stride를 "
              "줄이거나 total_sweeps를 늘리세요.")
        return []
    # 너무 촘촘하면 몇 개만 로그 스케일로 골라서 보여준다
    if len(good) > n_suggest:
        idx = np.linspace(0, len(good) - 1, n_suggest).astype(int)
        picked = sorted(set(good[i] for i in idx))
    else:
        picked = good
    print(f"  HIGH 등급 period 후보 (2의 거듭제곱): {good}")
    print(f"  추천 (로그 스케일로 {len(picked)}개 선택): {picked}")
    for p in picked:
        r = assess_te_reliability(p, meas_stride, total_sweeps)
        print(f"    period={p:5d}: samples/period={r['samples_per_period']:.1f}, "
              f"n_cycles={r['n_cycles']:.1f}")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-type", choices=["periodic", "telegraph"], default=None)
    ap.add_argument("--timescales", type=str, default=None)
    ap.add_argument("--dwells", type=str, default=None)
    ap.add_argument("--meas-stride", type=int, required=True)
    ap.add_argument("--total-sweeps", type=int, required=True,
                     help="보통 n_driven과 같은 값 (드라이브 구간 전체 sweep 수)")
    ap.add_argument("--power", type=float, default=0.01,
                     help="--signal-type telegraph 일 때 실제 tau_signal을 재기 위한 파워")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-suggest", action="store_true",
                     help="signal-type 없이, HIGH 등급 period를 자동으로 추천만 받는다")
    args = ap.parse_args()

    if args.auto_suggest:
        auto_suggest(args.meas_stride, args.total_sweeps)
        return

    if args.signal_type == "periodic":
        if not args.timescales:
            raise SystemExit("--signal-type periodic 에는 --timescales 가 필요합니다.")
        timescales = [float(x) for x in args.timescales.split(",")]
        results = run_periodic(timescales, args.meas_stride, args.total_sweeps)
    elif args.signal_type == "telegraph":
        if not args.dwells:
            raise SystemExit("--signal-type telegraph 에는 --dwells 가 필요합니다.")
        dwells = [float(x) for x in args.dwells.split(",")]
        results = run_telegraph(dwells, args.meas_stride, args.total_sweeps, args.power,
                                 args.seed)
    else:
        raise SystemExit("--signal-type periodic/telegraph 중 하나, 또는 --auto-suggest 를 "
                          "지정하세요.")

    overalls = [r["overall"] for _, r in results]
    print(f"\n{'='*64}")
    print(f"요약: HIGH={overalls.count('HIGH')}  MEDIUM={overalls.count('MEDIUM')}  "
          f"LOW={overalls.count('LOW')}  (전체 {len(overalls)}개)")
    low_vals = [v for (v, r) in results if r["overall"] == "LOW"]
    if low_vals:
        print(f"  LOW 등급(TE 분석 제외 권장): {low_vals}")
    print(f"{'='*64}")
    print("\nMethods 문장 예시:")
    print('  "Transfer entropy was evaluated only for parameter sets satisfying the')
    print('  protocol-check criteria (>=10 samples per period and >=50 cycles); phase lag')
    print('  and cross-correlation, which are far less sensitive to temporal aliasing, were')
    print('  retained for all parameter sets."')


if __name__ == "__main__":
    main()
