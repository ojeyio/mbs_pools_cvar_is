# loan_pool_generator.py
import numpy as np
import pandas as pd

# ---------- Helpers ----------
def monthly_payment(balance, annual_rate, remaining_months):
    if remaining_months <= 0:
        return 0.0
    r_m = annual_rate / 12.0
    if abs(r_m) < 1e-12:
        return balance / remaining_months
    return balance * (r_m * (1 + r_m) ** remaining_months) / ((1 + r_m) ** remaining_months - 1)

def generate_remaining_cashflows(remaining_balance, coupon, remaining_months, monthly_prepayment_rates):
    n = remaining_months
    cf = np.zeros(n)
    bal = remaining_balance
    payment = monthly_payment(bal, coupon, remaining_months)
    for t in range(n):
        if bal <= 0:
            break
        interest = bal * (coupon / 12.0)
        scheduled_prin = max(0.0, payment - interest)
        smm = monthly_prepayment_rates[t] if t < len(monthly_prepayment_rates) else monthly_prepayment_rates[-1]
        prepay = max(0.0, (bal - scheduled_prin) * smm)
        total_cf = interest + scheduled_prin + prepay
        cf[t] = total_cf
        bal = bal - scheduled_prin - prepay
    return cf[:t+1] if t+1>0 else np.array([])

def discount_factors_flat(rate_annual, n_months):
    t = np.arange(1, n_months + 1)
    return np.exp(-rate_annual * (t / 12.0))

# ---------- CPR model (simple) ----------
def annual_cpr_from_rate_diff(base_cpr, coupon, market_rate, rate_sensitivity=5.0, psa_multiplier=1.0):
    diff = market_rate - coupon
    lr = base_cpr * psa_multiplier * np.exp(-rate_sensitivity * diff)
    return max(0.0, min(1.0, lr))

def annual_cpr_vector(base_cpr, coupon, market_rate, term_months, psa_multiplier=1.0, ramp_months=30, rate_sensitivity=5.0):
    lr = annual_cpr_from_rate_diff(base_cpr, coupon, market_rate, rate_sensitivity, psa_multiplier)
    vec = np.ones(term_months) * lr
    for m in range(min(ramp_months, term_months)):
        vec[m] = lr * ((m + 1) / ramp_months)
    return vec

# ---------- CSV loader & pool builder (ADAPTED) ----------
def load_loan_csv_sample(path_csv, required_fields=None, n_sample=500, filters=None, random_state=42, column_map=None):
    """
    Load a loan-level CSV and sample loans for a pool.

    Args:
      - path_csv: path to CSV
      - required_fields: list of canonical keys we need (defaults used if None)
      - n_sample: how many loans to sample
      - filters: dict column->callable to filter rows (optional)
      - random_state: seed
      - column_map: OPTIONAL dict mapping canonical keys to your CSV column names, e.g.:
            {
             'orig_upb': 'LOAN_UPB',
             'current_upb': 'CURRENT_UPB',
             'orig_rate': 'INT_RATE',
             'orig_term': 'ORIG_TERM',
             'loan_age_months': 'LOAN_AGE'
            }
            If provided, mapping is used directly and heuristics are skipped.

    Returns:
      - df_sample: sampled dataframe (subset)
      - colmap: dictionary used for mapping canonical keys -> csv column name
    """
    df = pd.read_csv(path_csv, low_memory=True)

    # canonical field keys we want available
    canonical = ['orig_upb', 'current_upb', 'orig_rate', 'current_rate', 'orig_term', 'loan_age_months', 'orig_year']

    # If user provides mapping, validate
    if column_map:
        # ensure provided columns exist
        for k, v in column_map.items():
            if v not in df.columns:
                raise ValueError(f"column_map provided but '{v}' not found in CSV columns.")
        # build colmap filled with canonical keys mapped where possible
        colmap = {k: column_map.get(k, None) for k in canonical}
    else:
        # try heuristics to auto-detect common names
        lower_cols = {c.lower(): c for c in df.columns}
        colmap = dict.fromkeys(canonical, None)
        # heuristic matches
        heuristics = {
            'orig_upb': ['orig_upb','currentupb','current_upb','loanamount','original_upb','upb'],
            'current_upb': ['current_upb','curr_upb','currentupb','currupb'],
            'orig_rate': ['orig_rate','int_rate','loan_rate','interest_rate','intst'],
            'current_rate': ['current_rate','note_rate','coupon_rate'],
            'orig_term': ['orig_term','term','loan_term','original_term'],
            'loan_age_months': ['loan_age','loanage','age_months','age','months_elapsed','months'],
            'orig_year': ['orig_year','origination_year','year_orig','orig_year']
        }
        for key, candidates in heuristics.items():
            for cand in candidates:
                if cand in lower_cols:
                    colmap[key] = lower_cols[cand]
                    break

    # check we have at least orig_upb and orig_rate and orig_term mapped
    for needed in ['orig_upb', 'orig_rate', 'orig_term']:
        if colmap.get(needed) is None:
            raise ValueError(f"Could not map required column '{needed}'. Please provide a column_map with the CSV's column name for this field.")

    # apply filters if provided
    if filters:
        for col, func in filters.items():
            if col not in df.columns:
                continue
            df = df[ df[col].apply(func) ]

    df_sample = df.sample(n=min(n_sample, len(df)), random_state=random_state).copy()
    return df_sample, colmap

def build_pool_from_loans(df_sample, colmap, pool_params=None):
    """
    Construct per-loan remaining balances, coupons, remaining months, base CPR.
    Returns a list of loans: [{'balance':..., 'coupon':..., 'remaining_months':..., 'base_cpr':...}, ...]
    """
    loans = []
    default_base_cpr = pool_params.get('default_base_cpr', 0.06) if pool_params else 0.06
    for _, row in df_sample.iterrows():
        # balance: prefer 'current_upb' then fallback to orig_upb
        balance = None
        if colmap.get('current_upb') and colmap['current_upb'] in row.index:
            balance = row[colmap['current_upb']]
        if pd.isna(balance) or balance is None:
            balance = row[colmap['orig_upb']]

        # coupon (convert percent->decimal if necessary)
        coupon = row.get(colmap['orig_rate'])
        if pd.isna(coupon):
            coupon = 0.04
        else:
            coupon = float(coupon) / 100.0 if float(coupon) > 1.0 else float(coupon)

        # orig term detection
        term_val = row.get(colmap['orig_term'])
        if pd.isna(term_val):
            orig_term_months = 360
        else:
            term_val = float(term_val)
            if term_val > 100:
                orig_term_months = int(term_val)
            else:
                orig_term_months = int(term_val * 12)

        # loan age
        loan_age_months = 0
        if colmap.get('loan_age_months') and colmap['loan_age_months'] in row.index:
            try:
                loan_age_months = int(row[colmap['loan_age_months']])
            except Exception:
                loan_age_months = 0

        remaining_months = max(0, orig_term_months - loan_age_months)
        if remaining_months == 0:
            continue

        if pd.isna(balance):
            # fallback estimate
            balance = 100000.0 * (remaining_months / orig_term_months)

        base_cpr = pool_params.get('default_base_cpr', 0.06) if pool_params else 0.06
        loan = {'balance': float(balance), 'coupon': float(coupon), 'remaining_months': int(remaining_months), 'base_cpr': base_cpr}
        loans.append(loan)
    return loans

def loss_from_z_mbs_from_loans(z_array, loans, model_params=None, psa_multiplier=1.0, rate_sensitivity=5.0):
    """
    z_array: array-like standard normals -> mapped to parallel rate shocks
    loans: list of loan dicts as returned from build_pool_from_loans
    model_params: dict {r0, sigma, T}
    Returns: ndarray losses (len = len(z_array))
    """
    if model_params is None:
        model_params = {'r0': 0.03, 'sigma': 0.02, 'T': 1.0}
    r0 = model_params['r0']
    sigma = model_params['sigma']
    T = model_params.get('T', 1.0)

    # Precompute base PV of pool
    pv_base_total = 0.0
    loan_cache = []
    for loan in loans:
        rm = loan['remaining_months']
        cpr_vec = annual_cpr_vector(loan['base_cpr'], loan['coupon'], r0, rm, psa_multiplier=psa_multiplier, rate_sensitivity=rate_sensitivity)
        smm = 1.0 - np.power(1.0 - cpr_vec, 1.0 / 12.0)
        cf = generate_remaining_cashflows(loan['balance'], loan['coupon'], rm, smm)
        dfs = discount_factors_flat(r0, len(cf))
        pv = np.sum(cf * dfs)
        pv_base_total += pv
        loan_cache.append({'remaining_months': len(cf), 'coupon': loan['coupon'], 'balance': loan['balance'], 'base_cpr': loan['base_cpr']})

    losses = np.zeros(len(z_array), dtype=float)
    for i, z in enumerate(z_array):
        r_stressed = r0 + sigma * z * np.sqrt(T)
        pv_stressed_total = 0.0
        for lc in loan_cache:
            rm = lc['remaining_months']
            coupon = lc['coupon']
            cpr_vec = annual_cpr_vector(lc['base_cpr'], coupon, r_stressed, rm, psa_multiplier=psa_multiplier, rate_sensitivity=rate_sensitivity)
            smm = 1.0 - np.power(1.0 - cpr_vec, 1.0 / 12.0)
            cf = generate_remaining_cashflows(lc['balance'], coupon, rm, smm)
            dfs = discount_factors_flat(r_stressed, len(cf))
            pv_s = np.sum(cf * dfs)
            pv_stressed_total += pv_s
        losses[i] = pv_base_total - pv_stressed_total
    return losses
