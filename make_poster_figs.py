# make_poster_figs.py
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mbs_loss_generator import loss_from_z_mbs


def plot_variance_vs_N(summary_csv, outdir):
    df = pd.read_csv(summary_csv).sort_values('N')
    plt.figure(figsize=(8,5))
    plt.loglog(df['N'], df['cvar_mc_std'], marker='o', label='Naive MC (std error)')
    plt.loglog(df['N'], df['cvar_is_std'], marker='o', label='Static IS (std error)')
    plt.xlabel('Sample size (N)')
    plt.ylabel('Std error of CVaR estimate')
    plt.title('Std Error vs Sample Size (log-log)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    fn = os.path.join(outdir, 'variance_vs_N.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", fn)


def _load_array_any(path):
    try:
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr, dtype=float)
    except Exception:
        try:
            # try as text/csv
            arr = np.loadtxt(path, delimiter=',')
            return np.asarray(arr, dtype=float)
        except Exception:
            # last resort: allow pickle (if truly needed)
            arr = np.load(path, allow_pickle=True)
            return np.asarray(arr, dtype=float)


def plot_tail_histogram(losses_mc_path, losses_is_path, weights_path, outdir):
    losses_mc = _load_array_any(losses_mc_path)
    losses_is = _load_array_any(losses_is_path)
    weights = _load_array_any(weights_path)
    # normalize weights
    w_norm = weights / np.sum(weights)

    # full-range histogram
    plt.figure(figsize=(8,5))
    plt.hist(losses_mc, bins=200, density=True, alpha=0.45, label='Naive MC')
    plt.hist(losses_is, bins=200, weights=w_norm, alpha=0.35, label='Static IS (weighted)')
    plt.xlabel('Loss')
    plt.ylabel('Density / Weighted density')
    plt.title('Tail Histogram Overlay (Naive MC vs Weighted IS)')
    plt.legend()
    fn = os.path.join(outdir, 'tail_histogram.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", fn)

    # zoom on tail (90th to 100th percentile)
    lo = np.percentile(losses_mc, 90)
    hi = np.percentile(losses_mc, 100)
    plt.figure(figsize=(8,5))
    plt.hist(losses_mc, bins=200, range=(lo,hi), density=True, alpha=0.45, label='Naive MC')
    plt.hist(losses_is, bins=200, range=(lo,hi), weights=w_norm, alpha=0.35, label='Static IS (weighted)')
    plt.xlabel('Loss (tail)')
    plt.ylabel('Density')
    plt.title('Tail Histogram Overlay (90th–100th percentile)')
    plt.legend()
    fn2 = os.path.join(outdir, 'tail_histogram_tail_zoom.png')
    plt.savefig(fn2, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", fn2)


def plot_weights_histogram(weights_path, outdir):
    weights = _load_array_any(weights_path)
    plt.figure(figsize=(8,5))
    plt.hist(weights, bins=200)
    plt.yscale('log')
    plt.xlabel('Importance weight')
    plt.ylabel('Count (log scale)')
    plt.title('Importance Weights (log scale)')
    fn = os.path.join(outdir, 'weights_histogram.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", fn)


def stress_scenarios_plot(outdir, loans_or_pool_params, model_params, theta):
    # stress shifts to r0 for scenario comparison
    shifts = [0.0, 0.005, 0.01, 0.02]  # 0bp, 50bp, 100bp, 200bp
    cvars_mc = []
    cvars_is = []
    N = 50000
    for s in shifts:
        mp = dict(model_params)
        mp['r0'] = model_params['r0'] + s
        z_mc = np.random.randn(N)
        losses_mc = loss_from_z_mbs(z_mc, **loans_or_pool_params, model_params=mp)
        var_mc, cvar_mc = empirical_var_cvar(losses_mc, alpha=0.99)
        z_is = np.random.randn(N) + theta
        losses_is = loss_from_z_mbs(z_is, **loans_or_pool_params, model_params=mp)
        weights = np.exp(-z_is*theta + 0.5*theta**2)
        var_is, cvar_is = weighted_var_cvar(losses_is, weights, alpha=0.99)
        cvars_mc.append(cvar_mc)
        cvars_is.append(cvar_is)
    x = [f'+{int(100*s)}bp' for s in shifts]
    plt.figure(figsize=(8,5))
    plt.bar([i-0.15 for i in range(len(x))], cvars_mc, width=0.3, label='Naive MC')
    plt.bar([i+0.15 for i in range(len(x))], cvars_is, width=0.3, label='Static IS')
    plt.xticks(range(len(x)), x)
    plt.ylabel('Estimated CVaR (α=0.99)')
    plt.title('Stress Scenario CVaR Comparison')
    plt.legend()
    fn = os.path.join(outdir, 'stress_scenarios.png')
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", fn)

# helper small imports to reuse functions (copied from cvaar_experiments)
def empirical_var_cvar(losses, alpha=0.99):
    losses = np.sort(losses)
    n = len(losses)
    k = int(np.ceil(alpha * n)) - 1
    var = losses[k]
    tail = losses[k:]
    return var, tail.mean() if len(tail)>0 else var


def weighted_var_cvar(losses, weights, alpha=0.99):
    w = np.array(weights, dtype=float)
    losses = np.array(losses, dtype=float)
    wn = w / w.sum()
    order = np.argsort(losses)
    losses_s = losses[order]; wn_s = wn[order]
    cum = wn_s.cumsum()
    idx = np.searchsorted(cum, alpha, side='left')
    var = losses_s[idx] if idx < len(losses_s) else losses_s[-1]
    mask = losses >= var
    wtail = w[mask].sum()
    if wtail == 0:
        return var, var
    return var, (w[mask]*losses[mask]).sum() / wtail

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='output')
    p.add_argument('--theta', type=float, default=1.0)
    args = p.parse_args()
    outdir = args.outdir

    summary_csv = os.path.join(outdir, 'summary.csv')
    if not os.path.exists(summary_csv):
        raise SystemExit(f"summary.csv not found in {outdir}. Run cvaar_experiments.py first.")

    # Plot 1
    plot_variance_vs_N(summary_csv, outdir)

    # For Tail & Weights histograms the script expects:
    #   losses_mc.npy, losses_is.npy, weights.npy  (from weight_diag.py run)
    losses_mc_path = os.path.join(outdir, 'losses_mc.npy')
    losses_is_path = os.path.join(outdir, 'losses_is.npy')
    weights_path = os.path.join(outdir, 'weights.npy')
    if os.path.exists(losses_mc_path) and os.path.exists(losses_is_path) and os.path.exists(weights_path):
        plot_tail_histogram(losses_mc_path, losses_is_path, weights_path, outdir)
        plot_weights_histogram(weights_path, outdir)
    else:
        print("losses/weights files not found. Run the diagnostic script to create them:")
        print("  python3 weight_diag.py --N 100000 --theta <theta> --outdir", outdir)
        print("Then re-run this script to create tail and weight histograms.")

    # Final: stress scenarios (will re-simulate internally; requires loans_or_pool_params)
    # For this to work automatically we need to load the same loans/pool that generated the summary.
    # Quick heuristics: if you used loans previously, ensure a loans.pkl or the same column_map is available.
    print("Stress plot requires access to the loan pool; run a single stress script if you want that plot generated now.")
