cvaar_project/
├─ data/                         # put loan CSV(s) here (optional)
├─ output/                       # experiment outputs (CSV/PNG)
├─ cvaar_experiments.py          # main runner (MC vs static IS, with auto-calibration)
├─ loan_pool_generator.py        # load CSV, build loan pool
├─ mbs_loss_generator.py         # aggregate pool + wrapper for loans
├─ weight_diag.py                # save sample-level losses & weights for diagnostics
├─ make_poster_figs.py           # create poster-ready PNGs from outputs
├─ make_plots.py                 # (optional helper)
├─ requirements.txt
└─ README.md                     # this file

**MacOS**
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

**Window**
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

**1) Files & what they do (short)**

cvaar_experiments.py — main CLI. Runs naive MC and static exponential-tilt IS. Supports automatic θ calibration, accepts loan CSV input, saves per-repeat CSVs and summary.csv.

loan_pool_generator.py — helper to load a loan CSV, sample loans, build a pool representation.

mbs_loss_generator.py — aggregate-pool MBS loss generator (fast) and wrapper that calls loan-level generator when loans are provided.

weight_diag.py — runs one large trial and saves losses_mc.npy, losses_is.npy, weights.npy for plotting and diagnostics (ESS, weight hist).

make_poster_figs.py — produces three poster-ready PNGs (variance vs N, tail histogram, weights histogram) using outputs.

make_plots.py — minimal helper for plots if desired.

**2) How to run — synthetic (aggregate pool) — fastest for testing**
Run a quick smoke test (small numbers):
python3 cvaar_experiments.py --sample-sizes 5000 --repeats 5 --outdir output_test
Run a full synthetic sweep (poster numbers):
python3 cvaar_experiments.py \
  --calibrate --pilot-N 30000 \
  --sample-sizes 10000 20000 50000 100000 \
  --repeats 30 \
  --outdir output_synth

**What these do:**

If --calibrate is passed and --theta is not provided, the script runs a pilot and chooses a θ from the --theta-grid to target --target-tail fraction of proposal samples above the naive VaR (default ~0.08).

Output files: output_synth/results_N10000.csv, etc., and output_synth/summary.csv.


**3) How to run — real loan-level data (recommended for realistic results)**

A. Place your loan CSV in data/ (or provide full path). Inspect header row to find the column names for:

original/unpaid balance (UPB)

current UPB (optional)

note / coupon rate

original term (months or years)

loan age in months (optional)

**You can inspect headers quickly:**
head -n 1 data/your_loans.csv
# or use Python:
python3 - <<PY
import pandas as pd
df = pd.read_csv('data/your_loans.csv', nrows=0)
print(list(df.columns))
PY

B. Edit cvaar_experiments.py — find column_map_example and set the values to your CSV header names. Example:
column_map_example = {
  'orig_upb': 'ORIG_UPB',
  'current_upb': 'CURRENT_UPB',
  'orig_rate': 'NOTE_RATE',
  'orig_term': 'ORIG_TERM',
  'loan_age_months': 'LOAN_AGE'
}


C. Run the sweep (example):

python3 cvaar_experiments.py \
  --loan-csv data/your_loans.csv \
  --sample-loans 300 \
  --calibrate --pilot-N 30000 \
  --sample-sizes 10000 20000 50000 \
  --repeats 30 \
  --outdir output_real

Notes:

--sample-loans 300 will randomly sample 300 loans from your CSV to build a representative pool (reduces runtime). Increase if you want more diversity.

Use a larger --pilot-N for more robust θ calibration when using real pools.

**4) Diagnostics & Poster figures**

A. Run diagnostic (save sample-level losses & weights)
This produces losses_mc.npy, losses_is.npy, weights.npy used for tail/weights plots:

python3 weight_diag.py --N 100000 --theta <theta_from_calibration> --outdir output_real


If you used --calibrate earlier, the script printed the chosen θ (or check output_real/summary.csv).

B. Make poster-ready figures

python3 make_poster_figs.py --outdir output_real --theta <theta>


This creates:

variance_vs_N.png (log-log: std error vs sample size)

tail_histogram.png and tail_histogram_tail_zoom.png (full & tail zoom overlays)

weights_histogram.png (log-scale histogram of weights)

optionally, stress_scenarios.png if you re-simulate with access to the same loan pool

**5) Expected outputs & column meanings**

Per-run CSV: results_N{N}.csv — contains per-repeat values var_mc, cvar_mc, var_is, cvar_is.

Summary: summary.csv — one row per N with:

N — sample size

cvar_mc_mean, cvar_mc_std — mean & std (across repeats) for naive MC CVaR

cvar_is_mean, cvar_is_std — mean & std for IS CVaR

cvar_vrr — variance reduction ratio ≈ Var_mc / Var_is

**6) Tips, caveats & troubleshooting**
Theta calibration

--target-tail controls the fraction of proposal samples above naive VaR you want the pilot to target. Smaller target → more extreme θ.

If calibrated θ looks extreme or causes weight degeneracy, increase --pilot-N and widen --theta-grid or reduce --target-tail.

Too slow / big data

Public datasets are large. Sample a subset (e.g., 200–1000 loans) for poster experiments.

If you need faster inner loops at large N, request help to vectorize or numba-compile the generator.

**Common errors**

ModuleNotFoundError: No module named 'tqdm' → pip install tqdm or remove/replace tqdm usage.

RecursionError from earlier — resolved in current code (no recursion).

If Pandas cannot find columns in CSV, edit column_map_example to match exact header names.

Weight issues & ESS

After running weight_diag.py, compute ESS:

w = np.load('output_real/weights.npy')
ess = (w.sum()**2)/(w**2).sum()
print("ESS/N:", ess/len(w))


If ESS is very small (<< N), IS effective sample is small → consider smaller θ, mixture proposals, or alternative IS schemes.