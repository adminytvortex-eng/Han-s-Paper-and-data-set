"""
merge_fss_csv.py — 시드별 FSS 결과 CSV 합치기
============================================================
kagome_fss_v3.py가 시드별로 만든 fss_broad_L{L}_s{seed}.csv 파일들을
하나로 합쳐서 분석/논문용 데이터셋을 만듭니다.

사용법 (Windows CMD):
    python merge_fss_csv.py --dirs mining_L128 mining_L256

    # 출력 파일명 직접 지정하고 싶으면:
    python merge_fss_csv.py --dirs mining_L128 mining_L256 --out merged_all.csv

    # L128, L256을 각각 따로 합쳐서 따로 저장하고 싶으면:
    python merge_fss_csv.py --dirs mining_L128 --out merged_L128.csv
    python merge_fss_csv.py --dirs mining_L256 --out merged_L256.csv

동작:
  - 각 폴더에서 fss_*_L*_s*.csv 패턴의 파일을 모두 찾아서 합침
    (.state.npz 파일은 자동으로 무시됨)
  - 합친 후 (T, seed, L) 기준으로 중복 행 제거 (재실행으로 인한
    중복 라인이 있을 경우 안전장치)
  - L, seed, T 순으로 정렬
  - 시드 수, 온도점 수, 누락된 (L,seed) 조합이 있는지 등 요약 출력
============================================================
"""
import os
import sys
import glob
import argparse
import csv
from collections import defaultdict


def find_csv_files(dirs):
    """지정된 폴더들에서 fss_*.csv 패턴 파일 찾기 (.state.npz는 자동 제외)"""
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  ! 경고: 폴더 없음 — {d}")
            continue
        pattern = os.path.join(d, "fss_*_s*.csv")
        found = sorted(glob.glob(pattern))
        # .state.npz가 잘못 걸릴 일은 없지만 혹시 모를 .csv.state.npz류 제외
        found = [f for f in found if f.endswith(".csv")]
        print(f"  {d}: {len(found)}개 파일 발견")
        files.extend(found)
    return files


def merge_csv(files, out_path):
    if not files:
        print("합칠 CSV 파일이 없습니다. 폴더/경로를 확인하세요.")
        return

    all_rows = []
    fieldnames = None
    per_file_count = {}

    for fpath in files:
        with open(fpath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                print(f"  ! 경고: 컬럼이 다른 파일 발견 — {fpath}")
                print(f"      기존: {fieldnames}")
                print(f"      이파일: {reader.fieldnames}")
            rows = list(reader)
            per_file_count[fpath] = len(rows)
            all_rows.extend(rows)

    if fieldnames is None:
        print("유효한 CSV 헤더를 찾지 못했습니다.")
        return

    # 중복 제거: (L, seed, T) 기준 — 재실행으로 중복된 라인이 있을 경우 안전장치
    seen = set()
    dedup_rows = []
    n_dup = 0
    for row in all_rows:
        key = (row.get('L'), row.get('seed'), row.get('T'))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        dedup_rows.append(row)

    # 정렬: L, seed, T 순 (숫자로 변환해서 정렬 — 문자열 정렬 오류 방지)
    def sort_key(row):
        def to_f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return 0.0
        return (to_f(row.get('L')), to_f(row.get('seed')), to_f(row.get('T')))

    dedup_rows.sort(key=sort_key)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dedup_rows)

    # ── 요약 출력 ─────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  합치기 완료: {out_path}")
    print(f"{'='*55}")
    print(f"  입력 파일 수      : {len(files)}")
    print(f"  총 원본 행 수     : {len(all_rows)}")
    print(f"  중복 제거된 행 수 : {n_dup}")
    print(f"  최종 행 수        : {len(dedup_rows)}")

    # L별, seed별 통계
    by_L = defaultdict(set)
    by_L_T = defaultdict(set)
    for row in dedup_rows:
        L = row.get('L')
        seed = row.get('seed')
        T = row.get('T')
        by_L[L].add(seed)
        by_L_T[(L, seed)].add(T)

    print(f"\n  --- L별 요약 ---")
    for L in sorted(by_L.keys(), key=lambda x: float(x)):
        seeds = by_L[L]
        n_T_per_seed = [len(by_L_T[(L, s)]) for s in seeds]
        print(f"  L={L}: 시드 {len(seeds)}개  "
              f"(T점 수: min={min(n_T_per_seed)}, max={max(n_T_per_seed)})")
        if len(set(n_T_per_seed)) > 1:
            print(f"    ! 시드별로 T점 수가 다릅니다 — 일부 시드가 아직 "
                  f"덜 끝났거나 중간에 멈췄을 수 있습니다.")
            for s in sorted(seeds, key=lambda x: float(x)):
                n = len(by_L_T[(L, s)])
                if n != max(n_T_per_seed):
                    print(f"      seed={s}: {n}개 T만 완료")


def main():
    p = argparse.ArgumentParser(description="시드별 FSS CSV 합치기")
    p.add_argument('--dirs', nargs='+', required=True,
                    help='합칠 폴더 경로들 (예: --dirs mining_L128 mining_L256)')
    p.add_argument('--out', type=str, default='merged_fss.csv',
                    help='출력 파일명 (기본: merged_fss.csv)')
    args = p.parse_args()

    files = find_csv_files(args.dirs)
    merge_csv(files, args.out)


if __name__ == "__main__":
    main()
