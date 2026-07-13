"""
run_parallel_L256_fine.py — L=256 fine 스캔 (동시 6개)
============================================================
Windows CMD에서:
    python run_parallel_L256_fine.py

같은 폴더에 kagome_fss_v3.py가 있어야 합니다.

목적: L=48, L=128, L=256을 공정하게 비교하기 위해
      세 격자 모두 같은 fine grid(T=1.13~1.28, dT=0.005)로 재마이닝.

시드: 기존 L=256 30개 분석에 쓰인 시드(1,2,3,4,10~30)와 동일하게 맞춤
      (정찰 5개(5~9)는 fine에서는 제외 — 필요시 SEEDS에 추가)
============================================================
"""
import subprocess
import os
import sys
from multiprocessing import Pool

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

L = 256
MODE = "fine"
OUT_DIR = f"mining_L{L}_fine"
SCRIPT = "kagome_fss_v3.py"
N_PARALLEL = 6

T_MIN, T_MAX, DT = "1.13", "1.28", "0.005"  # L=128(1.195)과 L=48/256(1.21~1.23) 둘 다 포함

SEEDS = list(range(1, 31))  # 기존 L=256 30개 분석과 동일한 시드 (1~30 전체)


def run_one_seed(seed):
    log_dir = os.path.join(OUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"seed_{seed}.log")

    cmd = [
        sys.executable, SCRIPT,
        "--L", str(L),
        "--seed", str(seed),
        "--mode", MODE,
        "--T_min", T_MIN, "--T_max", T_MAX, "--dT", DT,
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
    print(f"L={L} fine 스캔: T={T_MIN}~{T_MAX} (dT={DT})")
    print(f"총 {len(SEEDS)}개 시드, 동시 최대 {N_PARALLEL}개씩 실행합니다.")
    print(f"시드 목록: {SEEDS}")
    print(f"출력: {OUT_DIR}/   로그: {OUT_DIR}/logs/\n")

    with Pool(processes=N_PARALLEL) as pool:
        results = pool.map(run_one_seed, SEEDS)

    failed = [s for s, code in results if code != 0]
    print(f"\n전체 {len(SEEDS)}개 시드 작업 종료.")
    if failed:
        print(f"  !! 오류난 시드: {failed}")
    else:
        print("  모두 정상 완료.")


if __name__ == "__main__":
    main()
