# make_plots.py
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_variance_vs_N(summary_csv, outdir):
    df = pd.read_csv(summary_csv)
    plt.figure(figsize=(6,4))
    plt.loglog(df['N'], df['cvar_mc_std'], marker='o', label='Naive MC (std error)')
    plt.loglog(df['N'], df['cvar_is_std'], marker='o', label='Static IS (std error)')
    plt.xlabel('Sample size (N)')
    plt.ylabel('Std error of CVaR estimate')
    plt.title('Std Error vs Sample Size')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.savefig(os.path.join(outdir, 'variance_vs_N.png'), dpi=300, bbox_inches='tight')
    print("Saved variance_vs_N.png")

def plot_tail_histogram(results_csv_N, outdir, theta):
    df = pd.read_csv(results_csv_N)
    # pick the first rep as example and load the losses? if per-rep CSV does not contain sample-level losses,
    # we'll approximate by re-simulating a large sample (fast fallback). For robust plots, run the script that saves sample-level losses.
    # Here we try to use a saved 'losses' CSV if present; otherwise user should re-run a single large trial that saves losses.
    print("Tail histogram plotting requires sample-level losses saved separately. If you have a losses file, place it in outdir and re-run.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='output')
    args = p.parse_args()
    summary_csv = os.path.join(args.outdir, 'summary.csv')
    plot_variance_vs_N(summary_csv, args.outdir)
