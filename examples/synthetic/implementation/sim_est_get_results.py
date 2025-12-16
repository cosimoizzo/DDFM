import time
import pandas as pd
import numpy as np

from synthetic_dgp.simulate import SIMULATE
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ as DFM
from models.ddfm import DDFM

import warnings

warnings.filterwarnings("ignore")


def run_sims(this_comb: tuple, n_mc_sim: int = 100) -> None:
    """
    Simulate n_mc_sim factor models, estimate and evaluate DFMs and DDFMs on them.
    Args:
        this_comb: configuration of the DGP for the factor model
        n_mc_sim: number of simulations (for each simulation, a new DGP is generated together with its data and on it a
            DFM and a DDFM are estimated and evaluated)

    Returns:
        None, it prints the results to a csv file with the name of the configuration taken from "this_comb".
    """
    dir_name_file = str(this_comb).replace(",", "").replace("(", "").replace(")", "").replace(" ", "_") + ".csv"
    r = this_comb[0]
    n_obs = this_comb[1]
    portion_missings = this_comb[2]
    n = this_comb[3]
    non_linear = this_comb[4]
    rho = this_comb[5]  # 0.7
    alpha = this_comb[6]  # 0.2
    u = 0.1
    tau = 0  # cross correlation among idio
    if non_linear:
        poly_degree = 2
        sign_features = r
    else:
        poly_degree = 1
        sign_features = 0

    results_dfm = np.zeros((n_mc_sim, 2))
    results_dfm_reduced_code_size = np.zeros((n_mc_sim, 2))
    results_ddfm = np.zeros((n_mc_sim, 3))
    results_ddfm_nnlin = np.zeros((n_mc_sim, 3))
    for n_mc in range(n_mc_sim):
        out = run_single_sim(seed=n_mc + 1, portion_missings=portion_missings,
                             n=n,
                             r=r, poly_degree=poly_degree,
                             sign_features=sign_features,
                             n_obs=n_obs,
                             rho=rho, alpha=alpha, u=u, tau=tau)
        results_dfm[n_mc, :] = out["results_dfm"]
        results_dfm_reduced_code_size[n_mc, :] = out["results_dfm_reduced_code_size"]
        results_ddfm[n_mc, :] = out["results_ddfm"]
        results_ddfm_nnlin[n_mc, :] = out["results_ddfm_nnlin"]
        del out
    df_results = pd.DataFrame(np.hstack((results_dfm,
                                         results_dfm_reduced_code_size,
                                         results_ddfm,
                                         results_ddfm_nnlin)),
                              columns=['dfm smoothed', 'dfm filtered', 'dfm elapsed time',
                                       'dfm code reduced size', 'dfm code reduced size vs linear f', 'dfm code reduced size elapsed time',
                                       'ddfm code', 'ddfm code reduced size', 'ddfm code reduced size vs linear f', 'ddfm elapsed time',
                                       'ddfm nnlin last neurons', 'ddfm nnlin code', 'ddfm nnlin code vs linear f', 'ddfm nnlin elapsed time'
                                       ])
    df_results.to_csv(dir_name_file)


def run_single_sim(seed: int, n: int = 10, portion_missings: float = 0, r: int = 2, poly_degree: int = 1,
                   sign_features: int = 0, n_obs: int = 50, rho: float = 0.7, alpha: float = 0.2, u: float = 0.1,
                   tau: float = 0, nnlin_decoder_ddfm_runs: int = 5) -> dict:
    """
    Simulate a single factor model, estimate and evaluate DFMs and DDFMs on it.
    Args:
        seed: seed setting for replicability
        n: number of observable components
        portion_missings: portion of missing data
        r: number of common factors
        poly_degree: polynomial degree (1 linear)
        sign_features: whether to add features based on sign (the first "sign_features" features, if 0 than not)
        n_obs: number of observations to simulate
        rho: parameter governing serial-correlation of the common factors
        alpha: parameter governing serial-correlation of the idiosyncratic components
        u: parameter governing the signal-to-noise ratio
        tau: parameter governing cross-correlation of the idiosyncratic components
        nnlin_decoder_ddfm_runs: number of runs for the nonlinear decoder to select best seed in terms
            of training loss

    Returns:
        a dictionary with the evaluation scores of the DFMs and the DDFMs.
    """
    results_dfm = np.zeros(3)
    results_dfm_reduced_code_size = np.zeros(3)
    results_ddfm = np.zeros(4)
    results_ddfm_nnlin = np.zeros(4)

    # simulate DGP
    sim = SIMULATE(seed=seed, n=n, r=r, poly_degree=poly_degree, sign_features=sign_features, rho=rho, alpha=alpha, u=u,
                   tau=tau)
    x = sim.simulate(n_obs, portion_missings=portion_missings)
    r_hat = sim.f.shape[1]

    # estimate dfm
    start_time = time.time()
    dyn_fact_mdl = DFM(pd.DataFrame(x), factors=min(r_hat, x.shape[1]), factor_orders=1)
    res_dyn_fact_mdl = dyn_fact_mdl.fit(disp=100)
    end_time = time.time()
    results_dfm[0] = sim.evaluate(res_dyn_fact_mdl.factors_ae.smoothed.values, f_true=sim.f)
    results_dfm[1] = sim.evaluate(res_dyn_fact_mdl.factors_ae.filtered.values, f_true=sim.f)
    results_dfm[2] = end_time - start_time

    # estimate dfm with reduced code size (aka number of factors)
    start_time = time.time()
    dyn_fact_mdl = DFM(pd.DataFrame(x), factors=min(r, x.shape[1]), factor_orders=1)
    res_dyn_fact_mdl = dyn_fact_mdl.fit(disp=100)
    end_time = time.time()
    results_dfm_reduced_code_size[0] = sim.evaluate(res_dyn_fact_mdl.factors_ae.smoothed.values, f_true=sim.f)
    results_dfm_reduced_code_size[1] = sim.evaluate(res_dyn_fact_mdl.factors_ae.smoothed.values, f_true=sim.linear_f)
    results_dfm_reduced_code_size[2] = end_time - start_time

    # estimate ddfm with linear decoder
    if poly_degree > 1:
        structure_encoder = (r_hat * 6, r_hat * 4, r_hat * 2, r_hat)
    else:
        structure_encoder = (r_hat,)
    start_time = time.time()
    deep_dyn_fact_mdl = DDFM(seed=seed, structure_encoder=structure_encoder, factor_order=1,
                             use_bias=False, link='relu', disp=100)
    deep_dyn_fact_mdl.fit(pd.DataFrame(x), build_state_space=False)
    end_time = time.time()
    results_ddfm[0] = sim.evaluate(np.mean(deep_dyn_fact_mdl.factors_ae, axis=0), f_true=sim.f)
    results_ddfm[3] = end_time - start_time
    # estimate ddfm with linear decoder but reduced code size and compare against nonlinear and linear factors
    # (only if nonlin dgp)
    if poly_degree > 1:
        # using same encoder structure of the symmetric autoencoder below
        structure_encoder = (r_hat, r * 9, r * 3, r)
        deep_dyn_fact_mdl = DDFM(seed=seed, structure_encoder=structure_encoder, factor_order=1,
                                 use_bias=False, link='relu', disp=100)
        deep_dyn_fact_mdl.fit(pd.DataFrame(x), build_state_space=False)
        results_ddfm[1] = sim.evaluate(np.mean(deep_dyn_fact_mdl.factors_ae, axis=0), f_true=sim.f)
        results_ddfm[2] = sim.evaluate(np.mean(deep_dyn_fact_mdl.factors_ae, axis=0), f_true=sim.linear_f)

    # estimate ddfm with non-linear decoder
    structure_encoder_nnlin = (r_hat, r * 9, r * 3, r)
    structure_decoder_nnlin = (r * 3, r * 9, r_hat)
    loss_now = None
    for j_seed in range(nnlin_decoder_ddfm_runs):
        start_time = time.time()
        deep_dyn_fact_mdl_nnlin = DDFM(seed=seed + j_seed,
                                       structure_encoder=structure_encoder_nnlin,
                                       factor_order=1,
                                       structure_decoder=structure_decoder_nnlin,
                                       use_bias=False, link='relu')
        deep_dyn_fact_mdl_nnlin.fit(pd.DataFrame(x), build_state_space=False)
        end_time = time.time()
        if loss_now is None or loss_now > deep_dyn_fact_mdl_nnlin.loss_now:
            loss_now = deep_dyn_fact_mdl_nnlin.loss_now
            last_neurons = np.mean(deep_dyn_fact_mdl_nnlin.last_neurons, axis=0)
            factors = np.mean(deep_dyn_fact_mdl_nnlin.factors_ae, axis=0)
            results_ddfm_nnlin[3] = end_time - start_time
    results_ddfm_nnlin[0] = sim.evaluate(last_neurons, f_true=sim.f)
    results_ddfm_nnlin[1] = sim.evaluate(factors, f_true=sim.f)
    results_ddfm_nnlin[2] = sim.evaluate(factors, f_true=sim.linear_f)

    # output dictionary
    out = {"results_dfm": results_dfm,
           "results_dfm_reduced_code_size": results_dfm_reduced_code_size,
           "results_ddfm": results_ddfm,
           "results_ddfm_nnlin": results_ddfm_nnlin,
           }
    return out
