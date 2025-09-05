# cvaar_experiments.py
"""
Main experiment runner for CVaR comparisons (naive MC vs static exponential-tilt IS).
Includes automated theta calibration (pilot-grid).

Example:
 python cvaar_experiments.py --calibrate --pilot-N 15000 --sample-sizes 10000 50000 --repeats 30 --outdir output
"""

import os
import argparse
import numpy as np
import pandas as pd
import time

# local imports
from loan_pool_generator import load_loan_csv_sample, build_pool_from_loans
from mbs_loss_generator import loss_from_z_mbs

np.random.seed(42)

# ---------------------
# Estimator utilities
# ---------------------
def empirical_var_cvar(losses, alpha=0.99):
    n = len(losses)
    k = int(np.ceil(alpha * n)) - 1
    sorted_losses = np.sort(losses)
    var = sorted_losses[k]
    tail = sorted_losses[k:]
    cvar = tail.mean() if len(tail) > 0 else var
    return var, cvar

def weighted_var_cvar(losses, weights, alpha=0.99):
    weights = np.array(weights, dtype=float)
    losses = np.array(losses, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights sum non-positive")
    w_norm = weights / total
    order = np.argsort(losses)
    sorted_losses = losses[order]
    sorted_w = w_norm[order]
    cum = np.cumsum(sorted_w)
    idx = np.searchsorted(cum, alpha, side='left')
    var = sorted_losses[idx] if idx < len(sorted_losses) else sorted_losses[-1]
    mask = losses >= var
    w_tail = weights[mask].sum()
    if w_tail == 0:
        cvar = var
    else:
        cvar = (weights[mask] * losses[mask]).sum() / w_tail
    return var, cvar

def normal_shift_weights(z_drawn_from_g, theta):
    z = np.array(z_drawn_from_g)
    return np.exp(-z * theta + 0.5 * theta**2)

# ---------------------
# Theta calibration (pilot)
# ---------------------
def calibrate_theta_via_pilot(theta_grid, pilot_N, alpha, target_tail_frac, loans_or_pool_params):
    """
    Choose theta from theta_grid by running a naive pilot (to estimate naive VaR)
    and then checking tail hit fraction when sampling from N(theta,1).
    loans_or_pool_params is the same dict passed to loss_from_z_mbs via ** expansion
    (it should already include 'model_params' or 'loans' / 'pool_params').
    """
    # naive pilot to get VaR under f
    z_pilot = np.random.randn(pilot_N)
    losses_pilot = loss_from_z_mbs(z_pilot, **loans_or_pool_params)
    var0, _ = empirical_var_cvar(losses_pilot, alpha=alpha)

    best_theta = None
    best_diff = float('inf')
    for th in theta_grid:
        z_g = np.random.randn(pilot_N) + th
        losses_g = loss_from_z_mbs(z_g, **loans_or_pool_params)
        tail_frac = (losses_g >= var0).mean()
        diff = abs(tail_frac - target_tail_frac)
        if diff < best_diff:
            best_diff = diff
            best_theta = th
    return best_theta, var0

# ---------------------
# Single trial / runner
# ---------------------
def run_single_trial(N, alpha, theta, loans_or_pool_params):
    # naive MC
    z_mc = np.random.randn(N)
    losses_mc = loss_from_z_mbs(z_mc, **loans_or_pool_params)
    var_mc, cvar_mc = empirical_var_cvar(losses_mc, alpha=alpha)

    # IS via normal-shift
    z_is = np.random.randn(N) + theta
    losses_is = loss_from_z_mbs(z_is, **loans_or_pool_params)
    weights = normal_shift_weights(z_is, theta)
    var_is, cvar_is = weighted_var_cvar(losses_is, weights, alpha=alpha)

    return var_mc, cvar_mc, var_is, cvar_is

def run_experiments(N, repeats, alpha, theta, loans_or_pool_params, outdir):
    rows = []
    for rep in range(repeats):
        var_mc, cvar_mc, var_is, cvar_is = run_single_trial(N, alpha, theta, loans_or_pool_params)
        rows.append({'rep': rep, 'N': N, 'var_mc': var_mc, 'cvar_mc': cvar_mc,
                     'var_is': var_is, 'cvar_is': cvar_is})
    df = pd.DataFrame(rows)
    fn = os.path.join(outdir, f'results_N{N}.csv')
    df.to_csv(fn, index=False)
    print(f"Saved per-repeat results to {fn}")
    return df

# ---------------------
# CLI + orchestration
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loan-csv', type=str, default=None)
    parser.add_argument('--sample-loans', type=int, default=300)
    parser.add_argument('--sample-sizes', nargs='+', type=int, default=[10000, 50000])
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--theta', type=float, default=None)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--pilot-N', type=int, default=20000)
    parser.add_argument('--target-tail', type=float, default=0.08)
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--theta-grid', nargs='+', type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # prepare dictionary to expand into loss_from_z_mbs(...)
    loans_or_pool_params = {}

    # if loan CSV given, build loans list and include it
    if args.loan_csv:
        column_map_example = {
            'loan_id': 'loan_id',
            'orig_upb': 'original_upb',
            'current_upb': 'current_upb',
            'orig_rate': 'note_rate',
            'orig_term': 'original_term_months',
            'orig_year': 'origination_year',
            'orig_ltv': 'orig_ltv',
            'fico': 'fico'
        }

        print("Loading loans from CSV (ensure column_map matches your CSV headers)...")
        df_sample, colmap = load_loan_csv_sample(args.loan_csv, n_sample=args.sample_loans, column_map=column_map_example)
        loans = build_pool_from_loans(df_sample, colmap, pool_params={'default_base_cpr': 0.06})
        print(f"Built loan pool with {len(loans)} loans (sampled).")
        loans_or_pool_params['loans'] = loans
    else:
        # default aggregate pool (fast fallback)
        pool_params = {'balance': 100000.0, 'coupon': 0.04, 'term_years': 30, 'base_cpr': 0.06, 'psa_multiplier': 1.0}
        loans_or_pool_params['pool_params'] = pool_params

    # model params included inside loans_or_pool_params so we don't pass model_params twice
    model_params = {'r0': 0.03, 'sigma': 0.02, 'T': 1.0}
    loans_or_pool_params['model_params'] = model_params

    # calibrate theta if requested and no explicit theta provided
    theta = args.theta
    if args.calibrate and theta is None:
        print("Calibrating theta via pilot runs...")
        chosen_theta, pilot_var0 = calibrate_theta_via_pilot(args.theta_grid, args.pilot_N, args.alpha, args.target_tail, loans_or_pool_params)
        theta = chosen_theta
        print(f"Calibrated theta = {theta} (naive pilot VaR = {pilot_var0:.6g})")

    if theta is None:
        theta = 1.5
        print(f"No theta supplied; using default theta = {theta}")

    # run experiments
    all_summaries = []
    start = time.time()
    for N in args.sample_sizes:
        print(f"\nRunning experiments for N={N} (repeats={args.repeats}) with theta={theta} ...")
        df = run_experiments(N, args.repeats, args.alpha, theta, loans_or_pool_params, args.outdir)
        s = {
            'N': N,
            'cvar_mc_mean': df['cvar_mc'].mean(),
            'cvar_mc_std': df['cvar_mc'].std(ddof=1),
            'cvar_is_mean': df['cvar_is'].mean(),
            'cvar_is_std': df['cvar_is'].std(ddof=1)
        }
        v_mc = s['cvar_mc_std']**2
        v_is = s['cvar_is_std']**2
        s['cvar_vrr'] = (v_mc / v_is) if v_is > 0 else None
        all_summaries.append(s)

    summary_df = pd.DataFrame(all_summaries).sort_values('N')
    summary_fn = os.path.join(args.outdir, 'summary.csv')
    summary_df.to_csv(summary_fn, index=False)
    end = time.time()
    print(f"\nSaved summary to {summary_fn}")
    print(summary_df)
    print(f"Total time: {end-start:.1f}s")

if __name__ == '__main__':
    main()
