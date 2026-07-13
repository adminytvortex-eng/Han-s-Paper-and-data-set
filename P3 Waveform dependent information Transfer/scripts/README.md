# Paper 3, Stage 1 — Lattice Response Spectroscopy

Most of the scripts in this repository are fully automated, eliminating the need for manual statistical analysis. Please note that many scripts output results in Korean for easier readability during the development process. I recommend using an AI-based tool to translate the code comments or output messages into your preferred language before running them.

While I have made every effort to document the workflow comprehensively, the simulation sequences were tailored to my specific research focus. If you need easy access to the raw data, please feel free to reach out to me directly; all raw datasets are archived on Google Drive

Code accompanying `Paper3_Stage1_Protocol_v1.docx`. Read the protocol document first —
it explains the design decisions (why K(t) bond-driving and not a pinning field, why
these three signals, why this output priority, etc.).

**Requires `kagome_vortex_core.py` (your file) in the same directory.** `kagome_lattice.py`
imports it directly for the actual lattice construction, vortex detection, and plaquette
adjacency graph — it is not reimplemented here. Put your copy of
`kagome_vortex_core.py` next to these files before running anything.

## Requirements

```bash
pip install numpy numba matplotlib networkx
```

(`networkx` is required transitively, by `kagome_vortex_core.build_kagome_lattice`.)

## Files

| File | Purpose |
|---|---|
| `kagome_lattice.py` | Thin wrapper around `kagome_vortex_core.py`. Run directly for a geometry self-test. |
| `signals.py` | Sine / White / Random-telegraph generators + power check. Run directly for a self-test. |
| `driven_kagome_sim.py` | Metropolis engine + all measurements. Run directly for a self-test (validates the H-map against a direct port of the reference formula, then runs a smoke-test drive). |
| `analyze_response.py` | Cross-correlation, transfer entropy, efficiency. Run directly for a self-test on synthetic data. |
| `run_stage1_experiment.py` | Top-level script — this is what you actually run for real results. |
| `checklist.py` | Interactive menu wrapper around everything below — see "Quick start (interactive)". |
| `pilot_calibration.py` | Measures equilibration time, autocorrelation times (baseline and driven), and driven-state transient length — run this before anything else. |
| `multiseed_stage1.py` | Runs all 3 cases across several seeds and reports Delta-Birth, absorbed energy, efficiency, and net TE(→Birth) as mean ± 95% CI — this is what actually tells you whether a single-seed pattern is real. |
| `merge_and_aggregate.py` | 따로따로 돌린 개별 시드 결과 + multiseed 배치 결과를 다시 실행하지 않고 합쳐서 하나의 seed-aggregated 리포트로 만든다. |
| `analyze_energy_vs_te.py` | Energy vs TE 산점도(Figure E), 입력 엔트로피 vs TE 산점도(Figure F)를 만들고, 케이스별 순위가 뒤바뀌는지 자동으로 확인한다. |
| `dwell_time_sweep.py` | Telegraph의 dwell_sweeps(부호 유지 시간)를 스윕 — entropy는 거의 고정한 채 상관시간만 바꿔서, TE가 정보량이 아니라 시간척도에 반응하는지 확인한다. |
| `square_wave_sweep.py` | 결정론적 사각파의 주기(period_sweeps)를 스윕 — dwell=1(random telegraph의 확률적 극한이 아니라 사실은 period=2 사각파)과 sine(period=40) 사이의 빈 주파수 구간을 채운다. |
| `protocol_check.py` | 새 입력 신호를 추가하기 전에 먼저 돌리는 나이퀴스트 사전 점검 -- 비싼 시뮬레이션 시작 전에 어떤 period/dwell 조합이 TE 분석에 못 쓰는지 미리 알려준다. |
| `recompute_tau.py` | Re-run the cross-correlation tau analysis on ALREADY-SAVED results with a different `--max-lag`, without rerunning the simulation. Regenerates the exact input signal from the saved seed/config and re-aligns it with the saved output CSV. |

## Quick start (interactive — recommended for a solo pilot)

```bash
python checklist.py
```

A menu instead of remembering CLI flags: shows/edits your current config (L, T, power, seed,
etc. — persisted in `checklist_config.json` between runs), and lets you run each step below
by picking a number. Same underlying scripts, just no retyping flags every time. Bumping the
seed (option 6) automatically points `outdir` at a fresh folder for the next repeat run.

## Quick start (manual — full CLI control)

0. **Calibrate first — measure equilibration and autocorrelation times, don't guess them.**
   ```bash
   python pilot_calibration.py --L 48 --T 0.9 --K0 0.7 --power 0.01
   ```
   Answers four questions before you commit to a production run: how many sweeps until the
   baseline equilibrates (N_THERM), how many sweeps apart two baseline measurements need to
   be to count as independent (τ_auto), how many sweeps after the drive switches on until the
   system settles into its new driven steady state (burn-in), and the same τ_auto question
   during driving. Prints a `RECOMMENDED SETTINGS` block at the end, including the exact
   `--burn-in-frac` value to pass to `run_stage1_experiment.py` (burn-in is stored as a sweep
   count internally but exposed as a *fraction* of `n_driven` on the CLI, since that's what
   `run_stage1_experiment.py` takes — the script does this division for you). Always look at
   the three saved plots (`baseline_equilibration.png`, `baseline_autocorrelation.png`,
   `driven_transient.png`) before trusting the printed numbers — the automatic detector is a
   starting point, not a substitute for your own judgement (see the plots' captions and the
   protocol document Section 5). Re-run with `--drive-case sine` or `--drive-case telegraph`
   if you want case-specific transient/τ_auto numbers instead of the White-noise default.

1. **Sanity-check every module** (each takes seconds):
   ```bash
   python kagome_lattice.py
   python signals.py
   python driven_kagome_sim.py
   python analyze_response.py
   ```
   All four should print "passed" / finite, sane-looking numbers with no errors.

2. **Linear-response check** (do this before trusting any real run):
   ```bash
   python run_stage1_experiment.py --check-linearity --L 24 --n-driven 20000
   ```
   Look at the printed `|Delta E|` ratio between the 2.0x-power and 0.5x-power runs (a 4x
   change in power P). In the linear-response regime, absorbed energy scales with *power*
   (not amplitude) — like Joule heating going as V² — so this ratio should come out close
   to **4x**, not 2x. (An earlier version of this check's message incorrectly said ~2x;
   if you see that old wording, ignore it — 4x is the correct expectation.)

3. **Run the real experiment**, now using your Step 0 measurements instead of guesses for
   `--n-therm`, `--meas-stride`, and `--burn-in-frac` (adjust `--L`, `--n-driven`, `--seed`
   for production scale; the values below are just illustrative):
   ```bash
   python run_stage1_experiment.py --L 48 --T 0.9 --K0 0.7 --power 0.01 \
       --n-baseline 5000 --n-driven 30000 --n-therm 2000 --meas-stride 5 \
       --burn-in-frac 0.05 --seed 0 --outdir stage1_results_seed0
   ```
   Repeat with different `--seed` values (≥5 seeds, per the protocol document Section 5)
   and average the summary CSVs across seeds before drawing conclusions.

4. Outputs land in `--outdir`:
   - `stage1_<case>_timeseries.csv` — raw per-sweep measurements for each case
   - `stage1_<case>_timeseries.png` — quick-look plot with the drive-start marked
   - `stage1_summary.csv` — one row per case: absorbed energy, efficiency, τ, transfer
     entropy, input entropy — this is the table for the paper
   - `stage1_comparison.png` — the three headline comparison plots

## After the pilot: checking reproducibility across seeds

A single seed showed two patterns worth checking before believing them: White noise
absorbs the most energy but has *negative* Delta-Birth (fewer vortices than baseline,
despite the most energy going in), while Sine absorbs the least energy but has by far the
largest net transfer entropy into Birth. Either pattern could be a real effect or a
single-seed fluke — the only way to tell is to look at several independent seeds.

```bash
python multiseed_stage1.py --seeds 0,1,2,3,4,5,6 --L 48 --T 0.9 --K0 0.7 --power 0.01 \
    --n-baseline 5000 --n-driven 30000 --n-therm 45 --meas-stride 5 \
    --burn-in-frac 0.0006 --outdir multiseed_results
```

(or use `checklist.py` option 8, which fills in your current config automatically). Produces:

- `multiseed_summary.csv` — every case x seed combination, one row each
- `delta_birth.png`, `absorbed_energy.png`, `efficiency.png`, `net_TE_birth.png` — mean ±
  95% CI (Student's t, appropriate for a handful of seeds) for each of the four quantities,
  one bar per case
- `all_four_figures.png` — the same four panels combined for a quick single glance
- A "HEADLINE CHECK" printed at the end that explicitly reports whether White's Delta-Birth
  and Sine's net TE(→Birth) are **confirmed negative/positive** (the entire CI on the
  correct side of zero) or **still cross zero** (not yet distinguishable from noise) — read
  this before anything else.

If both patterns hold up (CIs on the expected side of zero), the pilot's observation graduates
from "interesting single-seed curiosity" to "a reproducible effect worth building the next
paper section around." If either one doesn't, that's useful too — it tells you where to
spend the *next* round of compute (e.g. more seeds specifically for that case, or a closer
look at whether the effect depends on `power`, `L`, or the specific dwell/period parameters).

## 파일럿 이후 확장 실험: Energy vs TE, 그리고 dwell-time 스윕

7시드 재현성 확인 후 다음 두 가지를 추가로 확인했다:

**Figure E/F (`analyze_energy_vs_te.py`)**: Absorbed energy와 net TE(→Birth)를 산점도로
그려보면, White/Sine/Telegraph 세 케이스가 뚜렷하게 분리된 군집을 이룬다 — 에너지 흡수량
순위(white > sine > telegraph)와 TE 순위(sine > telegraph > white)가 서로 다르고, 특히
white와 telegraph의 순위가 완전히 뒤바뀐다. 입력 엔트로피 기준으로 봐도 마찬가지다. 즉
**"에너지 흡수량"과 "정보 전달량(TE)"은 서로 다른 축**이라는 것이 시각적으로, 그리고
순위 비교로 확인된다.

```bash
python analyze_energy_vs_te.py --summary merged_results/merged_summary.csv --outdir merged_results
```

**주의 **: 위 결과에서 "엔트로피보다 시간적 구조(structure)를 본다"까지
주장하려면 조심해야 한다 — white와 telegraph는 엔트로피 정의 자체와 상관시간이 동시에
다르기 때문에, 지금 세 점(sine/white/telegraph)만으로는 엔트로피 효과와 구조 효과를
분리할 수 없다. **엔트로피는 고정한 채 상관시간만 바꾸는 실험**이 필요하다.

**Dwell-time 스윕 (`dwell_time_sweep.py`)**: Telegraph 신호는 항상 ±A 두 값만 가지므로,
`dwell_sweeps`를 얼마로 바꾸든 입력 Shannon entropy는 거의 항상 1비트 근처로 고정된다.
즉 dwell_sweeps 하나만 스윕하면 "엔트로피는 고정, 상관시간만 변화"하는 실험이 자동으로
된다 — 새 신호를 설계할 필요가 없다.

```bash
python dwell_time_sweep.py --dwells 1,2,5,10,20,40 --seeds 0,1,2,3,4 \
    --L 48 --T 0.9 --K0 0.7 --power 0.01 --n-baseline 5000 --n-driven 30000 \
    --n-therm 45 --meas-stride 5 --burn-in-frac 0.0006 --outdir dwell_sweep_results
```

(또는 `checklist.py` 옵션 9). 결과물:
- `dwell_sweep_summary.csv` — dwell값 x 시드별 원본 데이터
- `dwell_sweep_TE_vs_correlation_time.png` — 3분할: (Sokal tau_auto vs dwell 파라미터,
  엔트로피가 실제로 고정되는지 확인, dwell에 따른 TE 변화)
- `dwell_sweep_TE_vs_tau_signal.png` — Sokal tau_auto 대 TE 참고용 산점도
- `dwell_sweep_TE_vs_dominant_period.png` — **메인 결과**: FFT 기반 지배주기(period) 대
  TE 산점도. dwell_sweeps가 작을수록(특히 1) 신호가 거의 매 스윕마다 부호를 바꾸는
  강한 반상관 신호가 되는데, 이때 Sokal의 적분 자기상관시간 공식은 `C(lag=1)≈-1`이 되어
  `tau = 0.5 + C(1) ≈ -0.5` 같은 음수로 깨진다 (계산 실수가 아니라 공식 자체가 느슨한
  상관을 가정하기 때문). 실제로 `dwell_sweeps=1`은 완전히 결정론적인 period-2 사각파가
  되어 이 현상이 정확히 나타난다 (실측: tau_signal=-0.500, 오차 없이 5개 시드 전부 동일).
  FFT 기반 지배주기는 이런 반상관 신호에서도 깨지지 않으므로, 모든 dwell값을 하나의
  일관된 축(주기, 스윕 단위)에서 비교하려면 이 그림을 봐야 한다. tau_signal이 음수인
  dwell값은 3분할 그림과 참고용 산점도에서 회색 X로 따로 표시된다.
- 이전 버전(FFT 주기 컬럼이 없는 CSV)으로 이미 돌려둔 결과가 있다면, 같은 outdir로
  다시 실행할 때 시뮬레이션을 다시 돌리지 않고 신호만 재생성해서 자동으로 컬럼을
  채워 넣는다(backfill) — 콘솔에 "backfill 완료" 메시지가 뜨면 정상.

TE가 dominant_period_fft에 따라 유의미하게 변한다면, "격자는 정보의 양이 아니라 정보의
시간척도(correlation time)에 반응한다"는 훨씬 절제되고 강한 결론으로 이어진다.

**그래프 폰트 관련 주의사항**: 이 두 스크립트는 콘솔 출력과 코드 주석은 한글이지만,
matplotlib 그래프 안의 제목/축/범례는 의도적으로 영어로 되어 있다 — 개발 중에 한글로
썼다가 기본 폰트가 한글 글리프를 지원하지 않아 그림에 빈 네모(□)로 깨져 나오는 걸
직접 확인했기 때문이다. 사용자 컴퓨터 환경에 따라 달라질 수 있는 문제라, 그래프 렌더링의
안정성을 위해 그림 안 텍스트만 영어로 유지한다.

## dwell=1은 random telegraph가 아니라 결정론적 사각파다 

`dwell_time_sweep.py`에서 발견한 `tau_signal=-0.5` 현상을 더 파보니, dwell_sweeps=1은
매 스윕마다 무조건 부호가 바뀌므로 확률적 요소가 전혀 없는 **결정론적 period-2
사각파**라는 것이 명확해졌다. 이걸 "이상하게 나온 telegraph 결과"로 다루지 않고,
`signals.py`에 `gen_square_wave()`라는 별도의 신호 계열로 명시적으로 분리했다:

- `dwell_time_sweep.py`의 결과 CSV에 이제 `signal_type` 컬럼이 있다 —
  `dwell=1`은 `"deterministic period-2 square wave"`, `dwell>=2`는
  `"random telegraph"`로 표시된다. 출력 테이블에도 이 분류가 그대로 나온다.
- **`square_wave_sweep.py`** (신규): 사각파의 `period_sweeps`를 자유롭게 스윕한다
  (`2,4,8,16,32,64,128,...`). dwell=1(=period 2)과 sine(period=40) 사이, 그리고 그
  너머의 주파수 구간을 채워서 TE(주기) 관계를 직접 확인할 수 있다. 사각파도 사인파처럼
  결정론적으로 주기적이므로, tau가 아니라 phase_lag로 응답을 측정한다
  (`run_stage1_experiment.py`의 `summarize_case`가 이제 `case in ("sine", "square")`일
  때 phase_lag 방식을 쓴다).

```bash
python square_wave_sweep.py --periods 2,4,8,16,32,64,128 --seeds 0,1,2,3,4 \
    --L 48 --T 0.9 --K0 0.7 --power 0.01 --n-baseline 5000 --n-driven 30000 \
    --n-therm 45 --meas-stride 5 --burn-in-frac 0.0006 --outdir square_sweep_results
```

(또는 `checklist.py` 옵션 10). `square_sweep_TE_vs_period.png`에 선형/로그 x축 두 버전이
같이 나온다 — period가 2부터 수백까지 넓게 퍼지므로 로그축이 보통 더 잘 보인다.

이렇게 하면 입력을 특징짓는 세 축 — **Shannon entropy(H)**, **correlation time(τ,
random telegraph 전용)**, **dominant frequency/period(ω, 사인파·사각파 전용)** — 을
독립적으로 스윕할 수 있게 되고, 최종적으로는 (H, τ, ω) → (TE, birth, drop)라는
응답 표면(response surface)을 그리는 연구로 이어진다.

## Paper 4를 위한 데이터 확장: 정보 필드의 "형태"를 지금부터 저장하기

지금까지는 정보를 `info`(=sqrt(mean(I^2))) 하나의 스칼라로 압축해서 봤다. 근데 정보밀도
필드 I(x,y)는 사실 훨씬 많은 걸 담고 있다 -- 평균/최댓값/분산/비대칭도/첨도, 공간
상관길이, 핫스팟이 하나의 큰 덩어리인지 여러 개로 흩어져 있는지 등. 지금(Paper 3)
분석에는 안 쓰더라도, Paper 4("파동이 정보의 양이 아니라 형태를 어떻게 바꾸는가")를
위해 지금부터 데이터를 풍부하게 저장해두면 나중에 다시 마이닝할 필요가 없다.

**옵션 1 -- 형태 스칼라 저장 (`--record-morphology`, 거의 공짜)**: 매 측정마다
`info_mean, info_max, info_std, info_skew, info_kurtosis, info_corr_length,
info_n_components, info_largest_component_frac`을 같이 저장한다. 이미 계산하고 있는
H-map(`compute_Hmap`)을 재사용하므로 추가 비용이 거의 없다.

```bash
python run_stage1_experiment.py ... --record-morphology
```

**옵션 2 -- 원본 필드 스냅샷 저장 (`--record-field-snapshots [--snapshot-stride N]`,
저장공간 필요)**: Persistence(hotspot이 시간에 따라 얼마나 유지되는지), Mutual
information(입력 신호와 필드 사이), Entropy production(dS/dt)은 한 시점의 스칼라만으론
계산할 수 없다 -- 여러 시점에 걸친 원본 필드가 있어야 나중에 후처리로 계산할 수 있다.
이 옵션을 켜면 `stage1_<case>_field_snapshots.npz`에 `(sweeps, fields)` 배열이 저장된다
(기본 저장 간격: `meas_stride`의 10배 -- 너무 촘촘하면 용량이 커지므로).

`checklist.py`의 옵션 0(설정 편집)에서 `record_morphology 1`, `record_field_snapshots 1`,
`snapshot_stride 20` 같은 식으로 켤 수 있다 (0=끔, 1=켬).

**지금 안 쓰는 것들 (Paper 4에서 스냅샷 데이터로 후처리할 예정)**:
- Persistence: 스냅샷 필드들에서 같은 hotspot을 시간에 따라 추적
- Mutual information (입력 ↔ 필드): 스냅샷의 각 셀 시계열과 delta_K(t)의 MI
- Entropy production: 스냅샷 간 필드 전체 엔트로피의 시간 변화율
- Information efficiency (ΔBirth/ΔInformation 등): 이미 저장된 `info`/`info_mean`과
  `birth`/`drop` 컬럼으로 지금도 계산 가능 (새 저장 불필요, `analyze_energy_vs_te.py`류
  스크립트를 참고해서 만들면 됨)

## TE의 temporal aliasing 문제 (v2: 3개 지표 x 3단계로 업그레이드)

Square-wave period 스윕에서 TE(net_TE_input_birth)가 period에 따라 부호까지 뒤집히는
현상이 발견됐다. 직접 재현 실험으로 확인:

```
period=16, meas_stride=5:  net_TE = -0.10  (음수)
period=16, meas_stride=1:  net_TE = +0.17  (양수!)
```

물리는 안 바뀌었는데 측정 간격만 바꿨더니 TE 부호가 뒤집혔다 -- temporal aliasing이다.
채찍GPT 피드백으로 처음엔 "period/stride >= 8" 단순 이분법을 썼다가, 다시 피드백을 받아
**3개 지표 x 3단계(FAIL/WARNING/PASS)** 체계로 업그레이드했다:

1. **Samples per period** (`timescale/meas_stride`) — 한 주기를 얼마나 세밀하게 쟀는가.
   FAIL<5, WARNING<10, PASS>=10.
2. **Number of cycles** (`total_sweeps/timescale`) — 그 주기가 몇 번이나 반복됐는가 (한
   주기를 잘 쟀어도 반복이 적으면 TE 추정 자체가 불안정하다). FAIL<20, WARNING<50,
   PASS>=50.
3. **tau_signal/meas_stride** (random telegraph류 전용, 실측값 사용) — 실측 상관시간이
   stride보다 느리면 WARNING.

종합 등급(`Expected TE reliability`)은 셋 중 가장 나쁜 것을 따른다: 하나라도 FAIL이면
`LOW`, WARNING만 있으면(FAIL 없이) `MEDIUM`, 전부 PASS면 `HIGH`.

**검증**: `total_sweeps=35000, meas_stride=5` 조건에서 이 시스템으로 직접 계산해보니,
채찍GPT가 손으로 추천한 period 목록(64, 96, 128, 192, 256, 512 = 전부 HIGH)과 "period=512
→ 68주기, period=2048 → 17주기"라는 계산까지 정확히 일치했다 (`assess_te_reliability`
함수가 실제로 68.4, 17.1을 반환).

**시스템 전체에 반영된 것**:
- `analyze_response.py`의 `assess_te_reliability()` — 위 3개 지표를 한 번에 계산.
  `check_nyquist_ratio()`는 레거시 이분법으로 남겨뒀다(하위호환).
- `run_stage1_experiment.py`의 `summarize_case()`가 매 실행마다 `te_reliability`
  ("HIGH"/"MEDIUM"/"LOW"), `signal_period_to_stride_ratio`, `te_n_cycles` 컬럼을 요약
  CSV에 추가한다. `white`처럼 뚜렷한 주기가 없는 신호는 이 체크 대상에서 제외되도록
  `signal_timescale_for_case()`가 `inf`를 반환하고, `assess_te_reliability()`도 이
  경우 곧바로 `HIGH`를 반환하게 특별 처리했다(안 그러면 `n_cycles=total/inf=0`이 되어
  버려서 엉뚱하게 FAIL 판정이 났었다 -- 실제로 겪은 버그).
- `square_wave_sweep.py`: 그래프에서 HIGH(파란 실선)/MEDIUM(주황 사각형)/LOW(회색 X)를
  3단계로 구분해서 그린다. 실험 시작 전에도 미리 등급을 계산해서 경고한다.
- **`protocol_check.py` v2**: 3개 지표를 전부 계산해서 ✅/⚠️/❌로 보여주고, `--auto-suggest`
  옵션으로 주어진 `meas_stride`/`total_sweeps`에서 HIGH 등급인 period를 2의 거듭제곱
  후보 중에서 로그 스케일로 자동 추천한다.

```bash
python protocol_check.py --signal-type periodic \
    --timescales 2,4,8,16,32,64,96,128,192,256,512,2048 \
    --meas-stride 5 --total-sweeps 35000

python protocol_check.py --signal-type telegraph --dwells 1,2,5,10,20,40 \
    --meas-stride 5 --total-sweeps 35000 --power 0.01   # tau_signal도 실제로 재서 판정

python protocol_check.py --auto-suggest --meas-stride 5 --total-sweeps 35000
```

(또는 `checklist.py` 옵션 11). 논문 Methods에는 이렇게 쓰면 된다: *"Transfer entropy was
evaluated only for parameter sets satisfying the protocol-check criteria (>=10 samples per
period and >=50 cycles); phase lag and cross-correlation, which are far less sensitive to
temporal aliasing, were retained for all parameter sets."*

**지금 실험(총 35000 sweep 기준) 추천 period**: 64, 96, 128, 192, 256, 512 (전부 HIGH).
2, 4, 8, 16은 phase_lag 전용으로 남기고 TE 그래프에서는 제외한다.

## Notes on the reference implementation (see protocol doc Section 7 for detail)

Three things were caught by the self-tests during development and are now resolved and
verified — not open issues, but worth knowing about if you extend this code:

- **Lattice/plaquette geometry**: an early draft of this code used a different, standalone
  bond-construction function (the one that only backs the H-map / `kagome_hmap_miner.py`
  screening pipeline, which has only L² real triangles — no "down" triangles) and mistakenly
  treated it as the physics lattice. This is now fixed: `kagome_lattice.py` imports
  `kagome_vortex_core.py` directly for bonds, the full 2×L² plaquette list, and the
  corner-sharing plaquette adjacency graph (verified there by networkx clique-finding, and
  by this repo's own self-test: BFS shell sizes 1,3,6,9 from any plaquette).
- **Do not round the winding number before histogramming for the H-map.** Winding numbers
  are exactly quantized to {-1,0,+1} in exact arithmetic, but the floating-point residual
  around 0 for non-vortex plaquettes has a sign that varies plaquette-to-plaquette, and the
  (-0.55,0.55) histogram has a bin edge exactly at 0 — so those residuals genuinely and
  correctly populate two different bins. Rounding erases this and collapses the whole H-map
  to a constant. `compute_Hmap()` in `driven_kagome_sim.py` reproduces the reference formula
  unrounded (checked line-for-line against a direct port of it, max abs diff = 0.0 in the
  module's self-test); vortex identification for birth/drop uses a separately-rounded copy
  of the same calculation, matching `kagome_vortex_core.detect_vortices` exactly.
- **Two legitimate information-density definitions coexist** in the reference codebase: a
  simple 3-category discrete entropy over the plaquette graph (screening-only,
  H_max=log₂3) and the 8-bin H-map used in the paper (H_max=log₂8=3). This code's Tier-2
  output is the latter.
- **Performance**: `kagome_vortex_core.mc_sweep` recomputes the trial energy via a full
  `phi.copy()` per site — fine for Paper 2's mining scale, too slow for Stage 1's many-sweep
  driven runs. `driven_kagome_sim.py` uses a Numba-jitted engine instead, with identical
  acceptance logic and proposal distribution; if you ever need bit-for-bit RNG-reproducible
  runs against the reference `mc_sweep`, use that function directly instead.

- **NaN handling in `drop`'s cross-correlation / transfer entropy (real bug, now fixed)**:
  `drop` is undefined (NaN) at any measurement step with zero vortices present. The
  original `cross_correlation_lag()` and `transfer_entropy()` did not handle this, so any
  NaN in `drop` silently poisoned the whole computation (mean/std/histogram all become NaN,
  and `np.argmax` on an all-NaN array returns a meaningless index — this is exactly what
  produced a `tau_input_to_drop` of `-50`, sitting right at the edge of the search window,
  in an early real run). Both functions now linearly interpolate over NaNs first (see
  `_fill_nan()` in `analyze_response.py`), with a printed warning if more than half the
  samples in a series are missing.
- **`tau` for a periodic (sine) drive is not a single well-defined number** — cross-correlating
  against a periodic input is itself periodic in lag, so the reported "tau" can jump around
  a lot depending on the search window (`--max-lag`) without indicating anything is wrong.
  Confirmed directly: the same run reported `tau=14` at `--max-lag 50` and `tau=101` at
  `--max-lag 150`, both "valid" local peaks within different cycles. Use `recompute_tau.py`
  to inspect the actual correlation-vs-lag curve for the sine case before reporting any
  single tau value for it; White noise and Random telegraph did not show this instability
  (their `tau` stayed at the same value regardless of `--max-lag`), consistent with them not
  having a dominant periodicity to alias against.

## Notes from the Stage-1 pilot run (methodology revisions — read before analyzing real data)

A single-seed pilot run surfaced two real analysis issues before any large-scale run was
committed to. Both are fixed in `analyze_response.py` / `run_stage1_experiment.py`:

- **`drop`'s cross-correlation / transfer entropy used to be silently corrupted by NaN.**
  `drop` is undefined (not just "missing") at any measurement step with zero vortices
  present. `cross_correlation_lag()` and `transfer_entropy()` now use **pairwise-valid**
  samples throughout — no NaN is ever interpolated or invented. For a given lag (or a given
  (Y_t, Y_t-1, X_t-1) triple for transfer entropy), only time points where every required
  quantity is simultaneously finite are used, and the actual sample count used is returned
  alongside every estimate (the `*_n_valid_*` columns in `stage1_summary.csv` /
  `recompute_tau.py`'s printed output). An earlier version linearly interpolated over the
  gaps instead, which is no longer done — Drop's domain of definition is genuinely "vortex
  present," and manufacturing values outside that domain is not appropriate for a result
  meant to go in a paper.
- **Sine's "response time" is not a single number under cross-correlation.** Cross-correlating
  a periodic input against anything is itself periodic in lag, so `argmax(cross-correlation)`
  does not define a τ for a periodic drive: the pilot run reported `τ=14` at `--max-lag 50`
  and `τ=101` at `--max-lag 150` — both "valid" local peaks in different cycles. The fix,
  implemented as `phase_lag_at_frequency()`, is a single-frequency (lock-in-amplifier-style)
  phase estimate at the known drive frequency ω instead — well-defined for any output
  amplitude, immune to this ambiguity, and the natural first step toward a real frequency
  scan later. `run_stage1_experiment.py` now reports a `phase_lag_input_to_<output>_rad`
  column for the Sine case instead of `tau_input_to_<output>` (which is left as `None`/NaN
  for Sine on purpose — do not backfill it).
- **Reliability tiers** for reporting in the paper (see protocol doc Section 6.6):
  - *High confidence*: absorbed energy, Birth efficiency, net TE(input→Birth), τ for
    White/Telegraph (confirmed stable across search-window size in the pilot).
  - *Conditional*: anything involving Drop — always report its `n_valid` alongside; TE for
    the Sine case.
  - *Method changed*: Sine's response is a phase lag φ(ω), not a τ.

## Notes on pilot_calibration.py's equilibration detector (development bugs, now fixed)

Two real bugs were caught and fixed while building the equilibration/transient detector —
both are the kind of thing that would have quietly produced misleading "recommended"
settings without any error message, so they're documented here in case you extend the code:

- **Naive standard errors badly underestimate scatter for autocorrelated MC data.** An
  early version compared block means using SEM = std/sqrt(N) with N = raw sample count. For
  correlated samples the true effective sample count is roughly N/(2·τ_auto), which can be
  many times smaller — using the naive (too-small) SEM made the consistency test far too
  strict. Fixed by explicitly correcting for τ_auto everywhere a standard error is computed.
- **Requiring every individual block to pass its own test is a multiple-comparisons trap.**
  A version that required each of ~20 blocks to *independently* pass a 2.5σ check against
  the reference kept reporting equilibration times of 75%+ of the run length, even for a
  series whose block means visibly had no trend at all — traced to a single noisy block
  failing its own test by chance anywhere late in the run poisoning every candidate cut point
  before it. Fixed by testing a single aggregate mean per candidate cut point instead
  (is *everything from this point on* consistent with the reference, as one comparison, not
  twenty). Also relatedly: Python's `x or default` silently replaces a **genuine and correct**
  result of `0` (e.g. "equilibrated immediately") with `default`, since `0` is falsy — fixed
  by using explicit `is not None` checks everywhere a detected value of exactly 0 is valid.

## Other open items



- **Telegraph dwell time**: currently a free parameter (`--dwell-sweeps`, default 20).
  Pick this so the telegraph spectrum is visibly different from white noise's flat
  spectrum (a quick FFT of the two signals is a good check before committing to a value).
- **Performance**: the Metropolis sweep is Numba-JIT'd but still single-threaded and
  O(N) in pure Python-level looping outside the inner sweep. For L=48 (N=6912 sites)
  and tens of thousands of sweeps × 3 cases × 5 seeds, expect run times of minutes to
  low tens of minutes per case depending on your machine — nothing like the scale of
  the full FSS mining in Paper 1, but plan accordingly. Parallelize across seeds with
  multiprocessing if needed (same pattern as the Paper 1/2 mining scripts).
