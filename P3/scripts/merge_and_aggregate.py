"""
지금까지 따로따로 돌린 결과들(개별 시드로 돌린 것 + multiseed_stage1.py로 배치로 돌린 것)을
합쳐서 하나의 seed-aggregated 리포트(4개 그림 + CSV)로 만드는 스크립트. 시뮬레이션을 다시
돌릴 필요 없음 -- 이미 저장된 stage1_summary.csv / multiseed_summary.csv들을 읽어서 합치기만
한다.

두 종류의 입력을 받는다:
  1. 개별 시드로 돌린 결과 폴더 (checklist.py의 옵션 4, 즉 run_stage1_experiment.py로 만든 것):
     안에 stage1_summary.csv와 stage1_run_config.json이 있음 (seed 컬럼은 없어서, 이
     config에서 seed를 읽어와 채워 넣는다)
  2. multiseed_stage1.py로 배치 실행한 결과 폴더:
     안에 multiseed_summary.csv가 바로 있고, 이미 seed 컬럼이 포함되어 있음

사용법:
    python merge_and_aggregate.py \\
        --single-seed-dirs stage1_results_seed0,stage1_results_seed1 \\
        --multiseed-dirs multiseed_results \\
        --outdir merged_results
"""

import argparse
import json
import os

import pandas as pd

from multiseed_stage1 import aggregate, print_summary_table, make_figures


def load_single_seed_dir(path):
    """stage1_summary.csv (seed 컬럼 없음) + stage1_run_config.json 을 읽어서 seed를 채운다."""
    summary_path = os.path.join(path, "stage1_summary.csv")
    config_path = os.path.join(path, "stage1_run_config.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} 가 없습니다 -- 경로를 확인해주세요.")
    df = pd.read_csv(summary_path)
    if "seed" not in df.columns:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"{summary_path}에 seed 컬럼이 없는데, {config_path}도 없어서 seed를 "
                f"알아낼 방법이 없습니다. --seed-override로 직접 지정해주세요.")
        with open(config_path) as f:
            cfg = json.load(f)
        df["seed"] = cfg["seed"]
    df["_source"] = path
    return df


def load_multiseed_dir(path):
    """multiseed_summary.csv 는 이미 seed 컬럼이 있으니 그대로 읽는다."""
    summary_path = os.path.join(path, "multiseed_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} 가 없습니다 -- 경로를 확인해주세요.")
    df = pd.read_csv(summary_path)
    df["_source"] = path
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-seed-dirs", type=str, default="",
                     help="콤마로 구분한 개별 시드 결과 폴더 목록 (예: "
                          "'stage1_results_seed0,stage1_results_seed1')")
    ap.add_argument("--multiseed-dirs", type=str, default="",
                     help="콤마로 구분한 multiseed_stage1.py 결과 폴더 목록 (예: "
                          "'multiseed_results')")
    ap.add_argument("--outdir", type=str, default="merged_results")
    args = ap.parse_args()

    if not args.single_seed_dirs and not args.multiseed_dirs:
        raise SystemExit("--single-seed-dirs 또는 --multiseed-dirs 중 최소 하나는 지정해야 "
                          "합니다.")

    os.makedirs(args.outdir, exist_ok=True)

    dfs = []
    if args.single_seed_dirs:
        for d in args.single_seed_dirs.split(","):
            d = d.strip()
            print(f"읽는 중 (개별 시드): {d}")
            dfs.append(load_single_seed_dir(d))
    if args.multiseed_dirs:
        for d in args.multiseed_dirs.split(","):
            d = d.strip()
            print(f"읽는 중 (multiseed 배치): {d}")
            dfs.append(load_multiseed_dir(d))

    merged = pd.concat(dfs, ignore_index=True)

    # 같은 (case, seed) 조합이 두 번 이상 들어왔는지 확인 -- 실수로 같은 시드를 두 번
    # 합치면 평균이 왜곡되므로 무조건 경고하고 중단한다.
    dup_mask = merged.duplicated(subset=["case", "seed"], keep=False)
    if dup_mask.any():
        print("\n경고: 같은 (case, seed) 조합이 중복으로 들어왔습니다 -- 계속하기 전에 "
              "확인해주세요:")
        print(merged.loc[dup_mask, ["case", "seed", "_source"]].sort_values(["case", "seed"])
              .to_string(index=False))
        raise SystemExit("\n중복을 제거하거나 입력 폴더 목록을 다시 확인한 뒤 재실행해주세요.")

    seeds_per_case = merged.groupby("case")["seed"].nunique()
    print(f"\n합쳐진 데이터: 케이스별 시드 개수:\n{seeds_per_case}")
    if seeds_per_case.nunique() > 1:
        print("주의: 케이스마다 시드 개수가 다릅니다 -- 원래 계획대로 모든 케이스에 같은 "
              "시드를 다 돌린 게 맞는지 확인해보시는 게 좋습니다.")

    merged_path = os.path.join(args.outdir, "merged_summary.csv")
    merged.to_csv(merged_path, index=False)
    print(f"저장: {merged_path} (총 {len(merged)}행)")

    metrics = ["delta_birth", "mean_absorbed_energy", "efficiency_birth_per_absorbedE",
               "net_TE_input_birth"]
    labels = ["Delta-Birth (raw)", "Absorbed energy", "Efficiency (Birth/AbsE)",
              "Net TE(input->Birth)"]
    agg = aggregate(merged, metrics)
    print_summary_table(agg, metrics, labels)

    print(f"\n그림 저장 위치: {args.outdir}/")
    make_figures(agg, args.outdir)


if __name__ == "__main__":
    main()
