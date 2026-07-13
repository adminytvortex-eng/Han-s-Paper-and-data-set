# Kagome XY Vortex Emergence — Paper 1 Dataset

Data supporting "[논문 제목]" (Han, Paper I of the Kagome XY series).

## Folder structure

```
kagome-xy-vortex-p1/
├── data/
│   ├── broad_scan/
│   │   ├── L48_seeds30.csv
│   │   ├── L64_seeds30.csv
│   │   ├── L96_seeds30.csv
│   │   ├── L128_seeds30.csv
│   │   └── L256_seeds30.csv
│   ├── fine_scan_dT0.005/
│   │   ├── L48_seeds30.csv
│   │   ├── L128_seeds30.csv
│   │   └── L256_seeds30.csv
│   └── L48_validation_seeds100/
│       └── L48_seeds100.csv
├── scripts/
│   └── (마이닝/분석에 쓴 스크립트들)
└── README.md
```

## Data description

- **`broad_scan/`**: Coarse temperature-grid scan across five lattice sizes
  (L=48, 64, 96, 128, 256) used to locate the approximate position of the
  η(T) peak. 30 independent seeds per lattice size.

- **`fine_scan_dT0.005/`**: Fine temperature-grid scan (ΔT = 0.005) at
  L=48, 128, 256, used to pin down the precise peak location identified in
  the broad scan. 30 independent seeds per lattice size.

- **`L48_validation_seeds100/`**: A separate run at L=48 with 100 seeds
  (rather than 30), used to validate that 30 seeds constitute a sufficient
  sample size for the statistics reported in the main text.

## Known data issue (L=128)

During the broad-scan stage, a subset of the original L=128 runs were
affected by a multiprocess data-contamination bug. These runs were
identified, discarded, and re-executed; the L=128 data included in this
repository (`broad_scan/L128_seeds30.csv` and
`fine_scan_dT0.005/L128_seeds30.csv`) reflect the corrected, final runs
only. No other lattice size was affected.

## Reproducing the main result

The peak temperature reported in the manuscript, T_η = 1.1789 ± 0.0001,
is obtained from `fine_scan_dT0.005/` via [방법: parabolic interpolation /
dη/dT zero-crossing / 등, 논문 방법론 절 참조].
