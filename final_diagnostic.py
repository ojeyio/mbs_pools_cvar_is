import numpy as np
import pandas as pd

# adjust paths to your output files
losses = np.load('output_diagnostic/losses_is.npy', allow_pickle=True)   # or path to your losses_is
weights = np.load('output_diagnostic/weights.npy', allow_pickle=True)   # or path to your weights

# normalize weights
w = weights.astype(float)
wn = w / w.sum()

# overall ESS
ESS = 1.0 / (wn**2).sum()
N = len(wn)
print(f"Overall ESS = {ESS:.1f}, ESS/N = {ESS/N:.3f} (N={N})")

# Weighted quantile function (returns VaR at prob alpha)
def weighted_quantile(values, weights_norm, q):
    order = np.argsort(values)
    v = values[order]
    w = weights_norm[order]
    c = w.cumsum()
    idx = np.searchsorted(c, q, side='left')
    idx = min(max(idx, 0), len(v)-1)
    return v[idx]

alpha = 0.99
VaR_w = weighted_quantile(losses, wn, alpha)   # weighted VaR
# tail mask (losses >= VaR_w)
mask_tail = losses >= VaR_w

W_tail = wn[mask_tail].sum()                    # total normalized weight in tail
ESS_tail = (W_tail**2) / (wn[mask_tail]**2).sum() if W_tail>0 else 0.0

# weighted CVaR (conditional average in tail)
if W_tail>0:
    CVaR_w = (wn[mask_tail] * losses[mask_tail]).sum() / W_tail
else:
    CVaR_w = float('nan')

print(f"Weighted VaR (alpha={alpha}) = {VaR_w:.2f}")
print(f"Weighted CVaR (alpha={alpha}) = {CVaR_w:.2f}")
print(f"Tail total normalized weight W_tail = {W_tail:.4f}")
print(f"Tail ESS = {ESS_tail:.1f}")
print(f"Naive expected tail count (MC) = {(1-alpha)*N:.1f}")

max_w = w.max()
median_w = np.median(w)
print("max/median weight ratio:", max_w/median_w)

