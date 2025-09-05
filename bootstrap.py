import pandas as pd
df = pd.read_csv('output_simulated_poster/results_N5000.csv')  # adjust path
print(df.columns)
mc = df['cvar_mc'].values
is_ = df['cvar_is'].values
# bootstrap CI of mean diff:
import numpy as np
rng = np.random.default_rng(1234)
diffs = is_ - mc
bs = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(5000)]
ci = np.percentile(bs, [2.5,97.5])
print("mean diff:", diffs.mean(), "bootstrap 95% CI:", ci)
