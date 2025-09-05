import numpy as np
import pandas as pd
from scipy import stats

# === ADJUST PATHS ===
losses_mc_path = 'output_diagnostic/losses_mc.npy'    # optional
losses_is_path = 'output_diagnostic/losses_is.npy'
weights_path    = 'output_diagnostic/weights.npy'
results_mc_csv  = 'output_simulated_poster/results_N5000.csv'   # per-repeat results file (if you used N=5000)
results_is_csv  = 'output_simulated_poster/results_N10000.csv'   # same file usually contains both cols

# === LOAD (allow_pickle True if needed) ===
losses_is = np.load(losses_is_path, allow_pickle=True)
weights   = np.load(weights_path, allow_pickle=True)

# normalize weights
w = np.asarray(weights, dtype=float)
if w.sum() == 0:
    raise ValueError("Weights sum to zero")
wn = w / w.sum()
N = len(wn)

# overall ESS
ESS = 1.0 / (wn**2).sum()
print(f"Overall ESS = {ESS:.1f}  ESS/N = {ESS/N:.3f}  (N={N})")

# simple weight stats
print("max weight:", w.max(), "median weight:", np.median(w), "max/median:", w.max()/max(np.median(w),1e-12))

# weighted VaR and CVaR utilities
def weighted_quantile(vals, weights_norm, q):
    order = np.argsort(vals)
    v = np.asarray(vals)[order]
    w = np.asarray(weights_norm)[order]
    c = np.cumsum(w)
    idx = np.searchsorted(c, q, side='left')
    idx = min(max(idx, 0), len(v)-1)
    return v[idx]

alpha = 0.99
VaR_w = weighted_quantile(losses_is, wn, alpha)
tail_mask = losses_is >= VaR_w
W_tail = wn[tail_mask].sum()
if W_tail > 0:
    CVaR_w = (wn[tail_mask] * losses_is[tail_mask]).sum() / W_tail
else:
    CVaR_w = np.nan

# tail ESS (effective sample size inside tail)
if W_tail > 0:
    ESS_tail = (W_tail**2) / (wn[tail_mask]**2).sum()
else:
    ESS_tail = 0.0

print(f"Weighted VaR (alpha={alpha}) = {VaR_w:.2f}")
print(f"Weighted CVaR (alpha={alpha}) = {CVaR_w:.2f}")
print(f"Tail normalized weight W_tail = {W_tail:.4f}")
print(f"Tail ESS = {ESS_tail:.1f}   (naive MC tail count ~ {(1-alpha)*N:.1f})")

# === Paired or independent test on per-repeat results ===
# If you have per-repeat CSV (results_N*.csv) that contains columns 'cvar_mc' and 'cvar_is', do:
try:
    df = pd.read_csv(results_mc_csv)
    mc_vals = df['cvar_mc'].values
    is_vals = df['cvar_is'].values
    # if repeats correspond (same index) then paired t-test is valid
    t_stat, pval = stats.ttest_rel(is_vals, mc_vals)
    print(f"Paired t-test: t = {t_stat:.3f}, p = {pval:.4f}")
    # Also show mean diff and CI using bootstrap
    diffs = is_vals - mc_vals
    mean_diff = diffs.mean()
    # simple bootstrap CI
    nboot = 5000
    bs = []
    rng = np.random.default_rng(12345)
    for _ in range(nboot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        bs.append(sample.mean())
    ci_low, ci_high = np.percentile(bs, [2.5, 97.5])
    print(f"Mean diff = {mean_diff:.2f}, bootstrap 95% CI = [{ci_low:.2f}, {ci_high:.2f}]")
except Exception as e:
    print("Could not run paired test from CSV:", e)
    print("If you don't have per-repeat CSV, collect per-repeat results and re-run the paired test.")
