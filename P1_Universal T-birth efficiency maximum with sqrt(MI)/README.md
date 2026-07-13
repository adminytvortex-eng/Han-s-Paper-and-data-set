# Kagome XY Vortex Emergence — Paper 1 Dataset

Data supporting "[A Universal Maximum in Vortex-Emergence Efficiency at the BKT Transition of the Kagome XY Model]" (Han, Paper I of the Kagome XY series).


## Data description

- **`broad_scan/`**: Coarse temperature-grid scan across five lattice sizes
  (L=48, 64, 96, 128, 256) used to locate the approximate position of the
  η(T) peak. 5 independent seeds per lattice size.

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
repository 

## Reproducing the main result

The peak temperature reported in the manuscript, T_η = 1.1789 ± 0.0001,
is obtained from `fine_scan_dT0.005/` via [방법: parabolic interpolation /
dη/dT zero-crossing / 등, 논문 방법론 절 참조].
