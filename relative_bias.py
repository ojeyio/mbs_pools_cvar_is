import pandas as pd, numpy as np
df = pd.read_csv('output_simulated_poster/results_N5000.csv')  # or N10000 file
mc = df['cvar_mc'].values
is_ = df['cvar_is'].values
R = len(df)
mc_mean = mc.mean(); is_mean = is_.mean()
mc_std = mc.std(ddof=1); is_std = is_.std(ddof=1)
rel_diff = (is_mean - mc_mean) / mc_mean
# t-like statistic for difference of means (approx)
stderr = np.sqrt(mc_std**2 + is_std**2) / np.sqrt(R)
tstat = (is_mean - mc_mean) / stderr
print(f"MC mean={mc_mean:.2f}, IS mean={is_mean:.2f}, rel_diff={rel_diff*100:.2f}%")
print(f"t-like stat={tstat:.2f}  (|t|>2 roughly significant at p~0.05)")
