# Model Explanations with Influence Analysis on the Titanic Dataset

> **Course project – COMP??? – *****Model explanations using Influence Analysis***

---

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Repository Layout](#repository-layout)
3. [Quick Start](#quick-start)
4. [Running the Experiments](#running-the-experiments)
5. [Interpreting the Results](#interpreting-the-results)
6. [Re‑training Without Top‑k Influencers](#re‑training-without-top‑k-influencers)
7. [Environment & Dependencies](#environment--dependencies)
8. [Dataset](#dataset)
9. [Citing & Acknowledgements](#citing--acknowledgements)
10. [License](#license)

---

## Project Motivation

Influence‑based explanation methods return, for every test prediction, the subset of training examples that most affected that prediction.\
The original assignment asks us to study two orthogonal challenges:

- **Stability** – How sensitive are the *top‑k* influence lists to small changes in data splits, hyper‑parameters or random seeds?
- **Consistency** – How similar are the influence lists produced by two different methods (First‑Order Influence Functions and TracIn)?

To answer these questions we implement a full experimental pipeline around the classic **Titanic survival prediction** task.

---

## Repository Layout

```
├── data/                     # Raw & processed Titanic CSVs
├── notebooks/
│   ├── 01_train_baseline.ipynb
│   ├── 02_compute_influences.ipynb
│   ├── 03_compare_lists.ipynb
│   └── 04_ablation_retrain.ipynb
├── src/
│   ├── datamodule.py         # Data loaders + preprocessing
│   ├── model.py              # Baseline TensorFlow model
│   ├── influence_first_order.py
│   ├── influence_tracin.py
│   ├── metrics.py            # Kendall Tau, Jaccard, etc.
│   └── utils.py
├── results/                  # Stored influence lists, plots, metrics
├── requirements.txt
└── README.md                 # (this file)
```

> **Tip:** All heavy computations (influence scores, retraining) cache their outputs in `results/` so you can reproduce plots without recomputing.

---

## Quick Start

```bash
# 1. Clone & create a fresh environment
$ git clone <repo‑url>
$ cd influence‑titanic
$ python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (CPU‑only by default)
$ pip install -r requirements.txt

# 3. Download the Titanic CSV (≈ 65 kB) – script will auto‑fetch if missing
$ python src/utils.py --download‑data

# 4. Train baseline model & store checkpoints
$ jupyter nbconvert --to notebook --execute notebooks/01_train_baseline.ipynb
```

After the baseline is trained, you can proceed with the influence analyses:

```bash
# First‑order influence scores for 128 test passengers
$ jupyter nbconvert --to notebook --execute notebooks/02_compute_influences.ipynb

# Rank‑correlation and overlap metrics
$ jupyter nbconvert --to notebook --execute notebooks/03_compare_lists.ipynb

# Remove top‑k from training set and re‑train (k = 5, 10, 20)
$ jupyter nbconvert --to notebook --execute notebooks/04_ablation_retrain.ipynb
```

All generated plots (score distributions, tail behaviour, Kendall Tau heatmaps, ROC/AUC curves, etc.) will appear in `results/plots/`.

---

## Running the Experiments

1. **Influence distribution visualisation**\
   For each explanation method we plot the (log‑scaled) influence scores of the top‑50 training points, highlighting the long‑tail behaviour.
2. **Stability analysis**\
   We repeat the training process **N = 5** times with different seeds and compute intra‑method Kendall Tau correlation → mean ± std summarises stability.
3. **Consistency analysis**\
   Cross‑method comparison on matched test points using:
   - Kendall/Kendall‑Star rank correlation
   - Jaccard overlap of the top‑k sets (k ∈ {5, 10, 20})
4. **Ablation study**\
   Remove the top‑k most influential passengers from the training set, retrain, and measure drop in Accuracy & AUC.

Each step is parameterised via command‑line flags inside the notebooks/scripts so you can adjust *k*, number of seeds, or the model architecture.

---

## Interpreting the Results

| Question                           | Expected Observation                                       |
| ---------------------------------- | ---------------------------------------------------------- |
| Long‑tail influence distribution?  | Steep decay then flat tail; magnitude differs per method.  |
| Stable within method?              | High Kendall Tau (> 0.8) indicates stability.              |
| Consistent across methods?         | Moderate Tau / < 50 % overlap suggests disagreement.       |
| Does removing top‑k hurt accuracy? | Small k → negligible; large k (≥ 20) shows measurable drop |

See `results/report.pdf` for a full narrative discussion and figures.

---

## Re‑training Without Top‑k Influencers

Running `04_ablation_retrain.ipynb` produces a table summarising model performance before and after ablating the influential records.  A helper script compiles these numbers into the report as LaTeX.

---

## Environment & Dependencies

- Python 3.10+
- TensorFlow 2.16 or newer
- Pandas, NumPy, SciPy
- scikit‑learn ≥ 1.3 (Kendall Tau)
- Matplotlib & Seaborn (for plots)
- `deel‑ai‑influenciae` (commit `v0.1.2`)
- tqdm, tyro (CLI)

Optional extras:

- CUDA‑enabled TensorFlow for faster training
- Dockerfile provided (`docker build -t influence‑titanic .`)

---

## Dataset

The Titanic passenger manifest is fetched from the public Pandas repository: [https://github.com/pandas‑dev/pandas/blob/main/doc/data/titanic.csv](https://github.com/pandas‑dev/pandas/blob/main/doc/data/titanic.csv)

We perform minimal preprocessing (one‑hot encoding of categorical columns, imputation of missing ages and fares) consistent with the original assignment.

---

## Citing & Acknowledgements

- Original assignment brief by **Paolo Missier** (University of Birmingham).
- Influence method implementations courtesy of **DEEL‑AI ‘Influenciae’**.
- Titanic dataset © RMS Titanic Inc.

If you build upon or publish results from this codebase, please cite the corresponding influence papers and the assignment brief.

---

## License

This project is released under the **MIT License** – see `LICENSE` for details.
