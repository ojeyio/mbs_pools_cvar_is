# mbs_loss_generator.py
import numpy as np

# This import is optional at runtime only when you actually use loan-level generator;
# keep it local inside the wrapper to avoid circular imports in interactive sessions.
try:
    from loan_pool_generator import loss_from_z_mbs_from_loans
    _HAS_LOAN_GENERATOR = True
except Exception:
    # loan_pool_generator may not be present or importable in some contexts
    _HAS_LOAN_GENERATOR = False

# -------------------------
# Original aggregate-pool functions (renamed to avoid collision)
# -------------------------
def scheduled_monthly_payment(balance, annual_coupon, term_years):
    """
    Compute fixed monthly mortgage payment for a fully-amortizing loan.
    """
    r_m = annual_coupon / 12.0
    n = int(term_years * 12)
    if r_m == 0:
        return balance / n
    payment = balance * (r_m * (1 + r_m) ** n) / ((1 + r_m) ** n - 1)
    return payment

def simulate_pool_cashflows(balance, coupon, term_years, monthly_prepayment_rates):
    """
    Simulate monthly cashflows (interest + scheduled principal + prepayment)
    given an array of monthly SMM prepayment rates (length = term_months).
    Returns arrays of cashflows and outstanding balances (both length term_months).
    """
    n = int(term_years * 12)
    payment = scheduled_monthly_payment(balance, coupon, term_years)
    cf = np.zeros(n)
    bal = balance
    balances = np.zeros(n)
    for t in range(n):
        if bal <= 0:
            break
        interest = bal * (coupon / 12.0)
        scheduled_principal = max(0.0, payment - interest)
        # Single-month SMM (prepayment fraction of outstanding after scheduled principal)
        smm = monthly_prepayment_rates[t] if t < len(monthly_prepayment_rates) else monthly_prepayment_rates[-1]
        prepay_amount = max(0.0, (bal - scheduled_principal) * smm)
        total_cf = interest + scheduled_principal + prepay_amount
        cf[t] = total_cf
        balances[t] = bal
        bal = bal - scheduled_principal - prepay_amount
    # trim trailing zeros if any
    valid = cf > 0
    return cf[valid], balances[valid]

def annual_cpr_from_rate_diff(base_cpr, psa_multiplier, coupon, market_rate, rate_sensitivity=5.0):
    """
    Simple, smooth CPR model: CPR = base_cpr * psa_multiplier * exp(-k * (market_rate - coupon))
    - if market_rate > coupon => prepayment incentive is lower (fewer refis) => CPR reduced
    - k (rate_sensitivity) controls responsiveness
    """
    diff = market_rate - coupon
    cpr = base_cpr * psa_multiplier * np.exp(-rate_sensitivity * diff)
    # clamp 0..1
    cpr = max(0.0, min(1.0, cpr))
    return cpr

def annual_cpr_vector_for_term(base_cpr, psa_multiplier, coupon, market_rate, term_years, ramp_months=30, rate_sensitivity=5.0):
    """
    Return an array of annual CPRs for each month of the term (size = term_months).
    This includes a PSA-style ramp for the first `ramp_months` months (linear up to the long-run PSA level).
    """
    term_months = int(term_years * 12)
    # long-run CPR from rate diff
    lr_cpr = annual_cpr_from_rate_diff(base_cpr, psa_multiplier, coupon, market_rate, rate_sensitivity)
    # PSA ramp: standard PSA multiplies the long-run by (month/30) for first 30 months
    cpr_vec = np.ones(term_months) * lr_cpr
    for m in range(min(ramp_months, term_months)):
        cpr_vec[m] = lr_cpr * ((m + 1) / ramp_months)
    return cpr_vec

def discount_factors_from_flat_rate(rate_annual, n_months):
    """
    Continuous-discount approximation per month: DF(t months) = exp(-rate_annual * t/12)
    You can replace with more accurate term-structure discounting if available.
    """
    t = np.arange(1, n_months + 1)
    dfs = np.exp(-rate_annual * (t / 12.0))
    return dfs

def aggregate_loss_from_z_mbs(z_array,
                             pool_params=None,
                             model_params=None,
                             horizon_months=None):
    """
    The original aggregate-pool MBS loss generator (renamed).
    Generate MBS portfolio losses for an array of standard-normal shocks z_array.
    Returns a numpy array of losses, same length as z_array.

    pool_params (dict):
        balance: aggregate notional (e.g., 100_000_000)
        coupon: annual mortgage coupon (e.g., 0.04)
        term_years: e.g., 30
        base_cpr: e.g., 0.06 (6% annual baseline)
        psa_multiplier: e.g., 1.0 (100% PSA)
    model_params (dict):
        r0: base annual market rate used for discounting (e.g., 0.03)
        sigma: volatility coefficient mapping z to rate shock (annualized)
        T: effective horizon used when scaling z -> rate shock (years, e.g., 1.0)
        rate_sensitivity: k in CPR formula
    horizon_months: if you want to limit PV to shorter horizon; otherwise full term used.
    """
    # defaults
    if pool_params is None:
        pool_params = {'balance': 100_000.0, 'coupon': 0.04, 'term_years': 30,
                       'base_cpr': 0.06, 'psa_multiplier': 1.0}
    if model_params is None:
        model_params = {'r0': 0.03, 'sigma': 0.02, 'T': 1.0, 'rate_sensitivity': 5.0}

    balance = pool_params['balance']
    coupon = pool_params['coupon']
    term_years = pool_params['term_years']
    base_cpr = pool_params['base_cpr']
    psa_mult = pool_params.get('psa_multiplier', 1.0)

    r0 = model_params['r0']
    sigma = model_params['sigma']
    T = model_params.get('T', 1.0)
    rate_sensitivity = model_params.get('rate_sensitivity', 5.0)

    n_months = int(term_years * 12)
    if horizon_months is not None:
        n_months = min(n_months, int(horizon_months))

    losses = np.zeros(len(z_array), dtype=float)

    # Precompute base PV (discounted at r0) using base CPR (market_rate = r0)
    base_cpr_vec = annual_cpr_vector_for_term(base_cpr, psa_mult, coupon, r0, term_years, rate_sensitivity=rate_sensitivity)
    # Convert annual CPR to monthly SMM: SMM = 1 - (1 - CPR)^(1/12)
    base_smm = 1.0 - np.power(1.0 - base_cpr_vec, 1.0 / 12.0)
    cf_base, balances_base = simulate_pool_cashflows(balance, coupon, term_years, base_smm)
    n_base = len(cf_base)
    dfs_base = discount_factors_from_flat_rate(r0, n_base)
    pv_base = np.sum(cf_base * dfs_base)

    # For each z, compute stressed PV with market rate = r0 + sigma * z * sqrt(T)
    for i, z in enumerate(z_array):
        r_stressed = r0 + sigma * z * np.sqrt(T)
        # CPR vector under stressed market rate
        cpr_vec = annual_cpr_vector_for_term(base_cpr, psa_mult, coupon, r_stressed, term_years, rate_sensitivity=rate_sensitivity)
        smm = 1.0 - np.power(1.0 - cpr_vec, 1.0 / 12.0)
        cf_stressed, balances_stressed = simulate_pool_cashflows(balance, coupon, term_years, smm)
        n_s = len(cf_stressed)
        dfs_s = discount_factors_from_flat_rate(r_stressed, n_s)
        pv_stressed = np.sum(cf_stressed * dfs_s)
        loss = pv_base - pv_stressed
        losses[i] = loss

    return losses

# -------------------------
# Unified wrapper: choose loan-level if available, otherwise aggregate
# -------------------------
def unified_loss_from_z(z_array, loans=None, pool_params=None, model_params=None):
    """
    Unified wrapper that calls either the loans-based generator
    (if 'loans' list supplied AND loan_pool_generator is available)
    or the aggregate-pool generator above.

    Call signature examples:
      unified_loss_from_z(z_array, loans=loans_list, model_params=model_params)
      unified_loss_from_z(z_array, pool_params=pool_params, model_params=model_params)
    """
    # If loans provided and loan-based generator available, delegate
    if loans is not None and _HAS_LOAN_GENERATOR:
        return loss_from_z_mbs_from_loans(z_array, loans, model_params=model_params)
    # Otherwise use aggregate pool generator
    return aggregate_loss_from_z_mbs(z_array, pool_params=pool_params, model_params=model_params)

# Backwards-compatible export name used by the experiment driver
loss_from_z_mbs = unified_loss_from_z
