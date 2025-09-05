# weight_diag.py
import os, argparse, numpy as np
import matplotlib.pyplot as plt
from mbs_loss_generator import loss_from_z_mbs
from cvaar_experiments import normal_shift_weights, empirical_var_cvar, weighted_var_cvar

def run_and_save(N, theta, outdir, model_params=None, loans_or_pool_params=None):
    os.makedirs(outdir, exist_ok=True)
    # draw naive samples
    z_mc = np.random.randn(N)
    losses_mc = loss_from_z_mbs(z_mc, **(loans_or_pool_params or {}))
    np.savetxt(os.path.join(outdir, 'losses_mc.npy'), losses_mc)

    # IS samples
    z_is = np.random.randn(N) + theta
    losses_is = loss_from_z_mbs(z_is, **(loans_or_pool_params or {}))
    weights = normal_shift_weights(z_is, theta)

    # suppose your arrays are losses_is (list/np.array), weights (list/np.array)
    losses_is = np.asarray(losses_is, dtype=float)   # ensure numeric dtype
    weights   = np.asarray(weights, dtype=float)


    np.save(os.path.join(outdir, 'losses_is.npy'), losses_is)
    np.save(os.path.join(outdir, 'weights.npy'), weights)

    np.savetxt(os.path.join(outdir, 'losses_is.csv'), losses_is, delimiter=',')
    np.savetxt(os.path.join(outdir, 'weights.csv'), weights, delimiter=',')

    # np.savez_compressed('output_diagnostic/diagnostics.npz', losses_is=losses_is, weights=weights)


    # compute ESS
    w = weights
    ess = (w.sum()**2) / ( (w**2).sum() )
    print("ESS:", ess, "ESS/N:", ess/len(w))

    # plot tail overlay
    plt.figure(figsize=(6,4))
    plt.hist(losses_mc, bins=200, density=True, alpha=0.5, label='Naive MC')
    plt.hist(losses_is, bins=200, weights=w/w.sum(), alpha=0.4, label='Static IS (weighted)')
    plt.xlim(np.percentile(losses_mc, 50), np.percentile(losses_mc, 99.9))
    plt.legend()
    plt.title('Tail histogram overlay (zoom)')
    plt.savefig(os.path.join(outdir, 'tail_histogram_diag.png'), dpi=300, bbox_inches='tight')
    print("Saved tail_histogram_diag.png")
    # weight histogram
    plt.figure(figsize=(6,4))
    plt.hist(weights, bins=200)
    plt.yscale('log')
    plt.title('Weights histogram (log scale)')
    plt.savefig(os.path.join(outdir, 'weights_histogram.png'), dpi=300, bbox_inches='tight')
    print("Saved weights_histogram.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=100000)
    p.add_argument('--theta', type=float, default=1.0)
    p.add_argument('--outdir', type=str, default='output_diag')
    args = p.parse_args()
    run_and_save(args.N, args.theta, args.outdir)
