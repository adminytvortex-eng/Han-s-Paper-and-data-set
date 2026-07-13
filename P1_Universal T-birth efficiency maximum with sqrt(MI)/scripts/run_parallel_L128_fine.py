"""
run_parallel_L128.py — 시드 111, 5~30을 "동시 최대 6개"로 실행하는 런처
============================================================
Windows CMD에서:
    python run_parallel_L128.py

같은 폴더에 kagome_fss_v3.py, kagome_bkt_v2.py가 있어야 합니다.

동작:
  - multiprocessing.Pool(processes=6)으로 동시 실행 개수를 정확히 6개로 제한
  - 시드 하나가 끝나면 풀이 자동으로 다음 시드를 투입 (큐 방식)
  - 각 시드는 kagome_fss_v3.py를 서브프로세스로 호출 (기존 체크포인트
    기능 그대로 작동 — 중간에 멈춰도 다시 이 스크립트를 실행하면
    끝난 시드/T는 건너뛰고 이어서 진행)
  - 로그는 mining_L128/logs/seed_{N}.log 에 개별 저장
============================================================
"""
import subprocess
import os
import sys
from multiprocessing import Pool

# Windows CMD에서 한글 print 시 UnicodeEncodeError(cp1252) 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

L = 128
MODE = "fine"
OUT_DIR = f"mining_L{L}_fine_v2"  # T=1.15~1.30 (실제 피크 포함) — 기존 fine과 분리
SCRIPT = "kagome_fss_v3.py"
N_PARALLEL = 6  # ← 동시 실행 개수 (발열 조절용, 필요하면 여기만 수정)

SEEDS = list(range(6, 31)) + [111, 222, 333, 444, 555]  # 6~30(25개) + 5개 = 30개


def run_one_seed(seed):
    log_dir = os.path.join(OUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"seed_{seed}.log")

    cmd = [
        sys.executable, SCRIPT,
        "--L", str(L),
        "--seed", str(seed),
        "--mode", MODE,
        "--T_min", "1.15", "--T_max", "1.30", "--dT", "0.005",
        "--out_dir", OUT_DIR,
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"  -> seed={seed} 시작")
    with open(log_path, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    status = "완료" if result.returncode == 0 else f"오류(code={result.returncode})"
    print(f"  [{status}] seed={seed}  (로그: {log_path})")
    return seed, result.returncode


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"총 {len(SEEDS)}개 시드, 동시 최대 {N_PARALLEL}개씩 실행합니다.")
    print(f"시드 목록: {SEEDS}")
    print(f"출력: {OUT_DIR}/   로그: {OUT_DIR}/logs/\n")

    with Pool(processes=N_PARALLEL) as pool:
        results = pool.map(run_one_seed, SEEDS)

    failed = [s for s, code in results if code != 0]
    print(f"\n전체 {len(SEEDS)}개 시드 작업 종료.")
    if failed:
        print(f"  !! 오류난 시드: {failed}  (해당 로그 파일을 확인하세요)")
    else:
        print("  모두 정상 완료.")


if __name__ == "__main__":
    main()
