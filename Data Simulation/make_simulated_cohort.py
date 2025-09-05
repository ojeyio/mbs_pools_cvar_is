#!/usr/bin/env python3
# make_simulated_cohort.py
import numpy as np, pandas as pd, argparse, os
np.random.seed(42)

def generate_cohort(n=300, year_min=2015, year_max=2019):
    # loan ids
    loan_id = np.arange(1, n+1)
    # orig year: sample uniformly in 2015-2019 but weighted towards 2016-2018
    years = np.random.choice(np.arange(year_min, year_max+1), size=n, p=[0.15,0.2,0.25,0.25,0.15])
    # term months = 360
    term = np.ones(n) * 360
    # original UPB: lognormal around 200k
    upb = np.round(np.random.lognormal(mean=12.2, sigma=0.8, size=n))  # ~200k median
    # current UPB: slightly lower than original (some seasoning)
    curr_upb = (upb * np.random.uniform(0.6, 1.0, size=n)).round()
    # note_rate: mixture 2.5-5% range, realistic around 3.5%-4.25%
    note_rate = np.clip(np.random.normal(loc=0.04, scale=0.006, size=n), 0.02, 0.06)
    # orig LTV: many between 60 and 97
    ltv = np.clip(np.random.beta(a=2.0, b=3.5, size=n) * 100, 40, 97)
    # fico: normal around 720, clip 300-850
    fico = np.clip(np.random.normal(loc=720, scale=40, size=n).astype(int), 300, 850)
    # occupancy
    occ = np.random.choice(['Owner Occupied', 'Second Home', 'Investor'], size=n, p=[0.85,0.05,0.10])
    # product / purpose
    product = np.random.choice(['30YR Fixed'], size=n)
    # assemble df
    df = pd.DataFrame({
        'loan_id': loan_id,
        'origination_year': years,
        'original_term_months': term.astype(int),
        'original_upb': upb.astype(int),
        'current_upb': curr_upb.astype(int),
        'note_rate': note_rate,
        'orig_ltv': np.round(ltv,2),
        'fico': fico,
        'occupancy': occ,
        'product': product
    })
    return df

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--out', default='pool_sample_simulated.csv')
    args = ap.parse_args()
    df = generate_cohort(n=args.n)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out, " rows=", len(df))
