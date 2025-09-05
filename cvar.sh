# in each output folder you should have results_N*.csv (per-repeat results)
python3 - <<'PY'
import pandas as pd, glob, os
folders = ['output_simulated_poster', 'output_final_theta_0.6']
for f in folders:
    files = glob.glob(os.path.join(f,'results_*.csv'))  # adjust pattern if different
    if not files:
        print(f, "no results CSV found")
        continue
    df = pd.concat([pd.read_csv(fn) for fn in files], ignore_index=True)
    print(f"\n=== {f} ===")
    print("repeats:", len(df))
    print("cvar_is mean/std:", df['cvar_is'].mean(), df['cvar_is'].std(ddof=1))
    print("cvar_mc mean/std:", df['cvar_mc'].mean(), df['cvar_mc'].std(ddof=1))
PY
