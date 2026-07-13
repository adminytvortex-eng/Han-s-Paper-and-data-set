"""
run_batch_L256_L48.py — L256(25 seeds) → L48(70 seeds) 연속 배치 마이닝
============================================================
Windows CMD에서:
    python run_batch_L256_L48.py

같은 폴더에 kagome_fss_v3.py가 있어야 합니다.

동작:
  1단계: L=256, 시드 [1,2,3,4,10~30] (25개), mode=broad, 동시 6개
         완료 후 →
  2단계: L=48,  시드 [31~100] (70개), mode=broad, 동시 6개
  - 각 단계는 자체적으로 동시 6개 제한 풀(Pool)로 처리
  - 1단계가 전부 끝나야 2단계가 시작됨 (순차)
  - 로그는 단계별 폴더에 개별 저장
  - kagome_fss_v3.py의 체크포인트 기능 그대로 작동 — 중간에 끊겨도
    이 스크립트를 다시 실행하면 끝난 시드/T는 건너뛰고 이어서 진행
============================================================
"""
import subprocess
import os
import sys
import time
from multiprocessing import Pool

# Windows CMD에서 한글 print 시 UnicodeEncodeError(cp1252) 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT = "kagome_fss_v3.py"
N_PARALLEL = 6  # 동시 실행 개수 (발열 조절용)
MODE = "broad"

# ── 단계 정의 ────────────────────────────────────────────
STAGES = [
    dict(name="L256", L=256, seeds=[1, 2, 3, 4] + list(range(10, 31)),
         out_dir="mining_L256_batch2"),
    dict(name="L48",  L=48,  seeds=list(range(31, 101)),
         out_dir="mining_L48_batch"),
]


def run_one_seed(args):
    L, seed, out_dir = args
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"seed_{seed}.log")

    cmd = [
        sys.executable, SCRIPT,
        "--L", str(L),
        "--seed", str(seed),
        "--mode", MODE,
        "--out_dir", out_dir,
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"  -> L={L} seed={seed} 시작")
    with open(log_path, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    status = "완료" if result.returncode == 0 else f"오류(code={result.returncode})"
    print(f"  [{status}] L={L} seed={seed}  (로그: {log_path})")
    return L, seed, result.returncode


def run_stage(stage):
    name = stage["name"]; L = stage["L"]; seeds = stage["seeds"]; out_dir = stage["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  단계 시작: {name}  (L={L}, 시드 {len(seeds)}개, 동시 {N_PARALLEL}개)")
    print(f"  출력: {out_dir}/   로그: {out_dir}/logs/")
    print(f"{'#'*60}\n")

    t0 = time.time()
    args_list = [(L, s, out_dir) for s in seeds]
    with Pool(processes=N_PARALLEL) as pool:
        results = pool.map(run_one_seed, args_list)
    elapsed = time.time() - t0

    failed = [(L, s) for L, s, code in results if code != 0]
    print(f"\n  단계 종료: {name}  (총 {elapsed:.0f}s)")
    if failed:
        print(f"    !! 오류난 (L,seed): {failed}")
    else:
        print(f"    모두 정상 완료 ({len(seeds)}개)")
    return failed


def main():
    print(f"총 {len(STAGES)}단계 순차 실행: "
          f"{' -> '.join(s['name'] for s in STAGES)}")

    all_failed = {}
    t_total0 = time.time()
    for stage in STAGES:
        failed = run_stage(stage)
        all_failed[stage["name"]] = failed

    print(f"\n{'#'*60}")
    print(f"  전체 배치 완료 (총 {time.time()-t_total0:.0f}s)")
    for name, failed in all_failed.items():
        status = "모두 정상" if not failed else f"!! {len(failed)}개 오류: {failed}"
        print(f"    {name}: {status}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
