# theta_diagnostics.py
import numpy as np, pandas as pd, glob, os

def load_arrays(outdir):
    # adjust names if your script saved differently
    losses_path = os.path.join(outdir, 'losses_is.npy')
    weights_path = os.path.join(outdir, 'weights.npy')
    # try .npy, then .npz, then .csv fallbacks
    def try_load(p):
        if os.path.exists(p):
            try:
                return np.load(p, allow_pickle=True)
            except Exception:
                pass
        pnpz = p.replace('.npy', '.npz')
        if os.path.exists(pnpz):
            z = np.load(pnpz)
            # pick first array if keys unknown
            if hasattr(z, 'files') and len(z.files)>0:
                return z[z.files[0]]
        pcsv = p.replace('.npy', '.csv')
        if os.path.exists(pcsv):
            return np.loadtxt(pcsv, delimiter=',')
        raise FileNotFoundError(f"Cannot load array at {p} or {pnpz} or {pcsv}")
    losses = try_load(losses_path)
    weights = try_load(weights_path)
    return losses, weights

def weighted_quantile(values, weights_norm, q):
    order = np.argsort(values)
    v = values[order]; w = weights_norm[order]
    c = np.cumsum(w)
    idx = np.searchsorted(c, q, side='left')
    idx = min(max(idx, 0), len(v)-1)
    return v[idx]

def summarize(outdir, alpha=0.99):
    losses, weights = load_arrays(outdir)
    w = np.asarray(weights, dtype=float)
    wn = w / w.sum()
    N = len(wn)
    ESS = 1.0 / (wn**2).sum()
    # weighted VaR/CVaR
    VaR_w = weighted_quantile(losses, wn, alpha)
    mask = losses >= VaR_w
    W_tail = wn[mask].sum()
    CVaR_w = (wn[mask] * losses[mask]).sum() / W_tail if W_tail>0 else np.nan
    ESS_tail = (W_tail**2) / (wn[mask]**2).sum() if W_tail>0 else 0.0
    return {
        'outdir': outdir,
        'N': N,
        'ESS': ESS,
        'ESS/N': ESS / N,
        'VaR_w': float(VaR_w),
        'CVaR_w': float(CVaR_w),
        'W_tail': float(W_tail),
        'ESS_tail': float(ESS_tail),
        'naive_tail_count': (1-alpha)*N
    }

if __name__ == '__main__':
    # adjust these to the actual output folders you created:
    outdirs = ['output_diagnostic_reduced_theta', 'output_simulated_poster', 'output_final_theta_0.6']
    rows = []
    for d in outdirs:
        try:
            rows.append(summarize(d))
        except Exception as e:
            print("Failed for", d, e)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv('theta_summary.csv', index=False)
    print("Saved theta_summary.csv")
