"""
7시드 결과가 재현성을 확보한 뒤, 다음 단계로 넘어가기 전에 확인해야 할 두 가지 그림.

Figure E: Absorbed energy (x) vs Net TE(input->Birth) (y)
    "에너지 흡수량"과 "정보 전달량"이 같은 축이 아니라는 걸 한눈에 보여주는 산점도.
    케이스별 평균±95% CI(굵은 점+에러바)와 개별 시드(작은 점)를 같이 그려서,
    평균만 봐서는 안 보이는 시드 간 산포도 확인 가능.

Figure F: Input Shannon entropy (x) vs Net TE(input->Birth) (y)
    "TE가 단순히 입력의 정보량(엔트로피)에 비례하는가, 아니면 시간적 구조를 구별하는가"를
    확인하는 그림. 입력 엔트로피 순서는 sine(~3.7) > white(~3.0) > telegraph(~1.0)인데,
    TE 순서는 sine > telegraph > white 이다 -- white와 telegraph의 순서가 엔트로피 기준과
    TE 기준에서 서로 뒤바뀐다는 것 자체가, "격자는 정보량이 아니라 정보의 시간적 구조에
    반응한다"는 주장의 직접적인 근거가 된다 (아래 correlation 출력에서도 확인 가능).

사용법:
    python analyze_energy_vs_te.py --summary merged_results/merged_summary.csv \\
        --outdir merged_results
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

CASES = ["sine", "white", "telegraph"]
CASE_COLORS = {"sine": "tab:blue", "white": "tab:orange", "telegraph": "tab:green"}


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (x.mean() if n else float("nan")), float("nan"), n
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    return m, stats.t.ppf(0.975, df=n - 1) * sem, n


def scatter_with_case_means(df, x_col, y_col, title, xlabel, ylabel, outpath):
    # 주의: 그래프 안의 글자(title/label/legend)는 영어로 둡니다. matplotlib 기본 폰트가
    # 한글 글리프를 지원하지 않아서, 한글로 쓰면 사용자 컴퓨터에서 빈 네모(□)로 깨져
    # 나옵니다 (실제로 이 스크립트 개발 중에 직접 확인한 문제입니다). 콘솔 print()
    # 출력이나 코드 주석은 한글이어도 문제없습니다 -- 이건 이미지 렌더링에만 해당하는
    # 제약입니다.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for case in CASES:
        sub = df[df["case"] == case]
        color = CASE_COLORS[case]
        # 개별 시드 (작은 점, 투명)
        ax.scatter(sub[x_col], sub[y_col], s=25, alpha=0.35, color=color)
        # 케이스 평균 +/- 95% CI (굵은 점 + 에러바)
        mx, cix, _ = mean_ci95(sub[x_col])
        my, ciy, n = mean_ci95(sub[y_col])
        ax.errorbar(mx, my, xerr=cix, yerr=ciy, fmt="o", markersize=11,
                    color=color, label=f"{case} (n={n})", capsize=4, elinewidth=1.5,
                    markeredgecolor="black", markeredgewidth=0.8)

    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"  저장: {outpath}")


def report_ordering(df, x_col, y_col, x_label, y_label):
    """케이스 평균 기준으로 x, y 순위를 비교해서, 순위가 뒤바뀌는 케이스 쌍이 있는지 확인."""
    means = {case: (mean_ci95(df.loc[df["case"] == case, x_col])[0],
                     mean_ci95(df.loc[df["case"] == case, y_col])[0]) for case in CASES}
    x_order = sorted(CASES, key=lambda c: means[c][0], reverse=True)
    y_order = sorted(CASES, key=lambda c: means[c][1], reverse=True)
    print(f"\n  {x_label} 순위 (큰 순): {' > '.join(x_order)}")
    print(f"  {y_label} 순위 (큰 순): {' > '.join(y_order)}")
    if x_order == y_order:
        print(f"  -> 두 순위가 완전히 같습니다: {y_label}가 {x_label}만으로 설명될 가능성.")
    else:
        print(f"  -> 두 순위가 다릅니다 (특히 순서가 뒤바뀐 쌍이 있다는 것 자체가, "
              f"{y_label}가 {x_label} 하나만으로는 설명되지 않는다는 증거입니다).")


def pearson_across_seeds(df, x_col, y_col, label):
    """전체 시드(케이스 구분 없이 다 합쳐서)로 피어슨 상관계수 계산 -- 참고용.
    케이스마다 x,y가 다른 군집을 이루기 때문에, 이 상관계수는 '전체적으로 x가 크면 y도
    크다'는 대략적인 경향만 보여줄 뿐, 케이스 내부의 관계를 말해주지는 않는다는 점에 주의.
    """
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        print(f"  {label}: 유효 샘플이 너무 적어 상관계수를 계산할 수 없습니다.")
        return
    r, p = stats.pearsonr(x[mask], y[mask])
    print(f"  {label}: 전체 시드 합산 Pearson r={r:+.3f} (p={p:.3g}, n={mask.sum()}) "
          f"-- 케이스 구분 없이 합친 값이라 참고용입니다, 위 그림에서 케이스별 군집을 "
          f"직접 보는 것이 더 정확합니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True,
                     help="merge_and_aggregate.py 또는 multiseed_stage1.py가 만든 "
                          "요약 CSV 경로 (merged_summary.csv 또는 multiseed_summary.csv)")
    ap.add_argument("--outdir", type=str, default=".",
                     help="그림을 저장할 폴더 (기본: 현재 폴더)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.summary)

    print(f"불러온 데이터: {len(df)}행 ({df['case'].nunique()}개 케이스, "
          f"케이스당 시드 {df.groupby('case')['seed'].nunique().to_dict()})")

    print("\n=== Figure E: Absorbed energy vs Net TE(input->Birth) ===")
    scatter_with_case_means(
        df, "mean_absorbed_energy", "net_TE_input_birth",
        title="Figure E: Energy absorption and information transfer (TE) are different axes",
        xlabel="Absorbed energy (<E_driven> - <E_baseline>)",
        ylabel="Net TE(input -> Birth)",
        outpath=os.path.join(args.outdir, "figE_energy_vs_TE.png"))
    report_ordering(df, "mean_absorbed_energy", "net_TE_input_birth",
                     "에너지 흡수량", "TE")
    pearson_across_seeds(df, "mean_absorbed_energy", "net_TE_input_birth",
                          "에너지 흡수량 vs TE")

    print("\n=== Figure F: Input Shannon entropy vs Net TE(input->Birth) ===")
    scatter_with_case_means(
        df, "input_shannon_entropy_bits", "net_TE_input_birth",
        title="Figure F: TE is not explained by input entropy alone",
        xlabel="Input Shannon entropy (bits)",
        ylabel="Net TE(input -> Birth)",
        outpath=os.path.join(args.outdir, "figF_entropy_vs_TE.png"))
    report_ordering(df, "input_shannon_entropy_bits", "net_TE_input_birth",
                     "입력 엔트로피", "TE")
    pearson_across_seeds(df, "input_shannon_entropy_bits", "net_TE_input_birth",
                          "입력 엔트로피 vs TE")

    print("\n결론 힌트: 입력 엔트로피 순서(sine > white > telegraph)와 TE 순서(sine > "
          "telegraph > white)에서 white와 telegraph의 순위가 서로 뒤바뀐다면, 이건 "
          "'TE가 단순히 입력 정보량에 비례한다'는 가설을 기각하는 직접적인 증거입니다 -- "
          "격자가 정보의 '양'이 아니라 '시간적 구조(스펙트럼 모양)'를 구별한다는 뜻입니다.")


if __name__ == "__main__":
    main()
