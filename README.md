# Transfer Learning from Lithium-Ion to Zinc-Ion Batteries

[![Paper](https://img.shields.io/badge/paper-preprint-blue)](https://github.com/Xueyang-Song/li-zn-battery-transfer-learning)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Preprint**: "Transfer Learning from Lithium-Ion to Zinc-Ion Batteries: Cross-Chemistry Capacity Prediction with Limited Target Data"
> *Xueyang Song, 2024*

## Key Finding

Cross-chemistry transfer from Li-ion to Zn-ion batteries exhibits a **critical crossover at N≈20 training cells**: below this threshold, Li-ion labels actively harm Zn-ion predictions (negative transfer); above it, selective feature transfer outperforms naive baselines. More importantly, while Selective MT-GP never beats GP-Direct on raw RMSE, it provides **3.2× better-calibrated uncertainty at N=2** (NLL: 1.07 vs. 3.44; coverage: 81% vs. 50.5% for a stated 90% credible interval). The primary value of cross-chemistry transfer lies in **uncertainty calibration**, not point accuracy.

## Architecture

```
Selective MT-GP Kernel:
  k_total = k_shared(exp_b, exp_b') × k_task(t, t')
           + k_private(log_ΔVar, log_ΔVar') × 𝟙[t = t' = Zn]

  ├── k_shared (Matérn-5/2):  exp_b — universal Arrhenius degradation kinetics
  │                            transferred via inter-task correlation kernel
  └── k_private (Matérn-5/2): log Δvar(dQ/dV) — Zn-ion only
                               phase transition dynamics not shared with Li-ion
```

## Results at a Glance

Median RMSE (cycles) across 200 Monte Carlo trials on 10 held-out Zn-ion cells:

| Model           | N=2  | N=5  | N=10 | N=20 | N=40 |
|-----------------|------|------|------|------|------|
| Selective MT-GP | 48.8 | 45.9 | 44.8 | **43.4†** | **41.0†** |
| Standard MT-GP  | 49.2 | 47.9 | 46.0 | 44.0 | 41.7 |
| GP-Direct       | **48.0** | **45.5** | **44.0** | **43.0** | **40.1** |
| Shuffled MT-GP  | 44.2 | 44.2 | 44.1 | 44.1 | 43.6 |

† Selective MT-GP statistically outperforms Shuffled (Wilcoxon, p<0.05).

**Uncertainty calibration at N=2** (90% credible interval target):

| Model           | Coverage | NLL  |
|-----------------|----------|------|
| Selective MT-GP | **81.0%** | **1.07** |
| GP-Direct       | 50.5%    | 3.44 |

## Repository Structure

```
li-zn-battery-transfer-learning/
├── config/
│   ├── __init__.py
│   └── settings.py            # Pydantic-based config (env vars override)
├── pipeline/
│   ├── extract/
│   │   ├── severson.py        # Severson/MATR + CALCE loader
│   │   └── batterylife.py     # BatteryLife ZN-coin loader
│   ├── transform/
│   │   ├── activation.py      # PELT-based activation detection
│   │   ├── features.py        # exp_b + Δvar(dQ/dV) extraction
│   │   └── normalize.py       # Log-transform + unified StandardScaler
│   ├── experiment/
│   │   ├── models.py          # GP-Direct, MT-GP, Selective MT-GP, Shuffled
│   │   └── runner.py          # Monte Carlo experiment runner
│   └── validate/
│       └── schemas.py         # Pydantic data schemas
├── utils/
│   ├── logging_config.py
│   ├── metrics.py             # RMSE, NLL, coverage
│   └── signals.py             # Savitzky-Golay + signal utilities
├── tests/
│   ├── test_activation.py
│   ├── test_features.py
│   └── test_models.py
├── output/
│   ├── paper/
│   │   └── main.tex           # Full LaTeX paper
│   ├── figures/               # PDF + PNG figures (fig2–fig5)
│   └── tables/
│       ├── table1_main_rmse.tex
│       └── table2_calibration.tex
├── features/
│   └── pipeline_metadata.json # Fixed test-set cell IDs + pipeline config
├── results/
│   └── final_experiment_results.json  # Canonical results
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

## Data Sources

> **Important**: Raw data files are NOT included in this repository (see `.gitignore`). Download them separately using the links below.

### Li-ion Source Data

**1. Severson/MATR Dataset** (Primary Li-ion source)
- **Citation**: Severson, K.A. et al. "Data-driven prediction of battery cycle life before capacity degradation." *Nature Energy* **4**, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8
- **Data portal**: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
- **GitHub (Q(V) array format used here)**: https://github.com/petermattia/revisit-severson-et-al
- **Size**: 124 LFP cells cycled to 80% capacity retention
- **Format**: 1000×99 Q(V) matrices (voltage interpolated to 1000 points), cycle-life labels
- **Place at**: `data/li_ion/severson/`

**2. CALCE Dataset**
- **Citation**: He, W. et al., CALCE Battery Research Group, University of Maryland
- **URL**: https://calce.umd.edu/battery-data
- **Size**: 8 LFP/NMC cells used for augmentation (CS2 series)
- **Place at**: `data/li_ion/calce/`

### Zn-ion Target Data

**3. BatteryLife Dataset** (Zn-ion target)
- **Citation**: Chen, C. et al. "A Dataset for Data-Driven Battery Lifetime Prediction." *arXiv* 2306.06063 (2023)
- **Zenodo DOI**: https://doi.org/10.5281/zenodo.7771988
- **Size**: 100 ZN-coin cells (Batches 1–3), aqueous ZnSO₄ electrolyte, NEWARE cycler
- **Format**: cycle-indexed discharge capacity (mAh)
- **License**: CC BY 4.0
- **Place at**: `data/zn_ion/batterylife/`

## Installation

```bash
git clone https://github.com/Xueyang-Song/li-zn-battery-transfer-learning
cd li-zn-battery-transfer-learning
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+, GPyTorch 1.11+

## Quickstart

```bash
# Run full pipeline (preprocess → feature extraction → experiments → figures)
python main.py --stage all --job-id run001

# Run only experiments (using pre-computed features in features/)
python main.py --stage experiment

# Run only preprocessing
python main.py --stage preprocess

# Override settings via environment variables
BATTERY_N_MC_TRIALS=50 BATTERY_N_TARGETS="[2,5,10]" python main.py --stage experiment
```

### Configuration

Settings are managed via `config/settings.py` (Pydantic-based). All settings can be overridden with environment variables prefixed `BATTERY_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `BATTERY_N_MC_TRIALS` | 200 | Monte Carlo trials per N |
| `BATTERY_N_TARGETS` | [2,5,10,20,40] | Training set sizes |
| `BATTERY_TEST_N_CELLS` | 10 | Fixed held-out test cells |
| `BATTERY_RANDOM_SEED` | 42 | Global random seed |

## Output

After running the pipeline, outputs are written to `output/`:

```
output/
├── paper/main.tex             # Full LaTeX source (compile with pdflatex)
├── figures/
│   ├── fig2_feature_distributions.{pdf,png}   # Li vs Zn feature distributions
│   ├── fig3_final_comparison.{pdf,png}         # RMSE vs N (4 models)
│   ├── fig4_mechanism_analysis.{pdf,png}       # Normalization failure modes
│   └── fig5_uncertainty_calibration.{pdf,png}  # Coverage + NLL vs N
└── tables/
    ├── table1_main_rmse.tex                    # LaTeX RMSE table
    └── table2_calibration.tex                  # LaTeX calibration table
```

To compile the paper:
```bash
cd output/paper
pdflatex main.tex && pdflatex main.tex
```

## Tests

```bash
python -m pytest tests/ -v  # 47 tests across activation, features, models
```

Tests cover: PELT activation detection, feature extraction (exp_b, dQ/dV), GP model training, MC runner, metrics (RMSE/NLL/coverage).

## Citation

```bibtex
@article{song2026transfer,
  title={Transfer Learning from Lithium-Ion to Zinc-Ion Batteries: Cross-Chemistry Capacity Prediction with Limited Target Data},
  author={Song, Xueyang},
  year={2026},
  note={Preprint}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

Note: Data files (BatteryLife, Severson/MATR, CALCE) are governed by their respective licenses. BatteryLife is CC BY 4.0. Severson/MATR and CALCE data should be downloaded directly from their respective portals.
