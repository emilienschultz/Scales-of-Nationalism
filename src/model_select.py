import pandas as pd
import numpy as np
import warnings

from joblib import Parallel, delayed
from kneed import KneeLocator
from scipy.stats import chi2, chi2_contingency
from sklearn.base import clone
from src.model_fit import build_latent_model, do_StepMix, do_kmeans, do_AHC, do_hdbscan
from src.tooling import MIN_CLUSTER_SHARE, min_cluster_size
from stepmix.stepmix import bootstrap



##### Gap statistic #####

def bootstrap_gap(data, controls, n, model, params, iter_num):
    # Create a random dataset. StepMix's categorical measurement integer-casts
    # floats, so the categorical-latent null draws on [min, max + 1) to floor
    # to uniform integer categories including the top one. Everything else
    # (distance models, continuous latent) runs on continuous data, where that
    # +1 would inflate the reference box a full unit above the observed range.
    categorical = model == 'latent' and 'categorical' in params.get('msrt', '')
    high = data.max(axis=0) + 1 if categorical else data.max(axis=0)
    rand_data = np.random.uniform(
        low=data.min(axis=0),
        high=high,
        size=data.shape)
    # StepMix truncates floats to integer categories, so floor the categorical
    # null explicitly: the CVIs are then computed on the same integer data the
    # model was fitted on (as for the real data), not on the continuous draw.
    if categorical:
        rand_data = np.floor(rand_data)
    rand_data = pd.DataFrame(rand_data, columns=data.columns)
    
    # Fit the model
    if model == 'latent':
        res = do_StepMix(
            rand_data,
            controls = None, # if params.get('covar') == 'with' else None,
            n = n,
            **params)
    elif model == 'kmeans':
        res = do_kmeans(rand_data, n, **params)
    elif model == 'AHC':
        res = do_AHC(rand_data, n, **params)
    
    # The gap statistic only consumes the CVIs: drop the label vector here, in
    # the worker, so grid size x gap_iters partitions are never sent back or
    # held in memory (about 1 GB at the full-replication settings).
    res.pop('pred_clust', None)

    # Add iteration number
    res = pd.DataFrame([res])
    res['bootstrap_iter'] = iter_num + 1
    
    return res


# Indices where a LOWER score means better clustering. They keep Tibshirani's
# original orientation (gap = log(rand) - log(model), like within-dispersion);
# higher-is-better indices are flipped so that, for every index, a larger gap
# uniformly means "more structure than the uniform null" and get_gap's
# selection rule has the same semantics across indices.
LOWER_IS_BETTER = {'davies_bouldin'}


def compute_gap(bootstrap_results, model_results, model, params, indices):
    gap_values = pd.DataFrame()

    grouped = bootstrap_results.groupby('n_clust')
    
    for n_clust, group in grouped:
        # Get corresponding model score
        mod_scores = model_results.loc[
            (model_results['model'] == model) & 
            (model_results['params'] == params) &
            (model_results['n_clust'] == n_clust)
        ]
        
        row_data = {
            'model': model,
            'params': params,
            'n_clust': n_clust
        }
        
        # Calculate gap statistic for each index
        for index in indices:
            rand_ind = group[index]
            mod_ind = mod_scores[index]
            
            # Rescale Silhouette index if needed
            if index == 'silhouette':
                rand_ind = (rand_ind + 1) / 2
                mod_ind = (mod_ind + 1) / 2
            
            # Calculate gap statistic and s value, oriented per index so a
            # larger gap always means more structure than the null
            if index in LOWER_IS_BETTER:
                gap = np.log(np.mean(rand_ind)) - np.log(mod_ind)
            else:
                gap = np.log(mod_ind) - np.log(np.mean(rand_ind))
            s = np.std(np.log(rand_ind)) * np.sqrt(1 + (1 / len(group)))
            
            # Add to row data
            row_data[f'{index}_gs'] = gap.values[0]
            row_data[f'{index}_s'] = s
        
        # Append to results
        gap_values = pd.concat([gap_values, pd.DataFrame([row_data])], ignore_index=True)
    
    return gap_values


def get_gap(gap_values, model, params, index, sizes=None, min_size=None):
    """Optimal n for one (model, params) and one index, or 'none'.

    Rule (diverges from the notebooks and from the earlier versions here):

    1. a k is admissible only if its gap exceeds the spread of the null,
       gap(k) > s(k): the fit must score better than fits on uniform data.
       Without this the gap was only read for its shape, so a k was returned
       even on structureless data (all gaps negative);
    2. when `sizes` (smallest cluster size, indexed by n_clust) and `min_size`
       are given, a k is admissible only if its smallest cluster has at least
       `min_size` members. This removes the outlier splits (829 / 1) that beat
       the box null only because the null has no tails;
    3. among the admissible k, return the smallest one whose gap is within one
       s of the highest gap: gap(k) >= gap(k*) - s(k*), k* = argmax (the
       "globalSEmax" rule of R's cluster::maxSE). Tibshirani's original rule
       (smallest k with gap(k) >= gap(k+1) - s(k+1)) assumes a gap rising up
       to the true k, which holds for the within-dispersion but not for the
       validity indices: with 4 well separated classes the index often dips
       at k=3, and that rule then stopped at k=2.

    Returns 'none' when no k is admissible.
    """
    rows_id = ((gap_values['model'] == model) & (gap_values['params'] == params))
    df = gap_values[rows_id].sort_values('n_clust').reset_index(drop=True)

    gap = df[f'{index}_gs'].astype(float)
    s = df[f'{index}_s'].astype(float)

    # NaN gaps (k=1, undefined indices) compare False and drop out here
    ok = gap > s
    if sizes is not None and min_size is not None:
        ok &= df['n_clust'].map(sizes).ge(min_size)
    if not ok.any():
        return 'none'

    i_max = gap[ok].idxmax()
    threshold = gap[i_max] - s[i_max]
    return int(df.loc[ok & (gap >= threshold), 'n_clust'].min())


CVI_COLUMNS = ('silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn')


def select_candidates(all_models, gap_values, min_share=MIN_CLUSTER_SHARE,
                      indices=CVI_COLUMNS):
    """Gap-selected (model, params, n_clust) rows, flagged per index.

    One row per selected fit, with its sizes and index values copied from
    `all_models` and a `<index>_gap` column set to 1 for every index that
    selected it (NaN otherwise). Uses only stored outputs (`gap_values`,
    `all_models`), so it can be re-run on a cached result to apply the current
    selection rule or another `min_share` without re-fitting anything.

    `min_share` is the smallest admissible cluster, as a share of the sample
    (see `get_gap` and `tooling.min_cluster_size`). The frame is empty when no
    configuration beats the uniform null.
    """
    min_size = min_cluster_size(all_models, min_share)
    kept = ['min_clust_size', 'max_clust_size', *indices]
    rows = {}

    configs = gap_values[['model', 'params']].assign(
        _p=gap_values['params'].astype(str)
    ).drop_duplicates(['model', '_p'])

    for model, config in zip(configs['model'], configs['params']):
        fits = all_models[
            (all_models['model'] == model) & (all_models['params'] == config)
        ]
        sizes = fits.set_index('n_clust')['min_clust_size']
        for index in indices:
            best_n = get_gap(gap_values, model, config, index, sizes, min_size)
            if best_n == 'none':
                continue
            key = (model, str(config), best_n)
            if key not in rows:
                fit = fits[fits['n_clust'] == best_n].iloc[0]
                rows[key] = {'model': model, 'params': config, 'n_clust': best_n,
                             **{c: fit[c] for c in kept}}
            rows[key][f'{index}_gap'] = 1

    columns = ['model', 'params', 'n_clust', *kept,
               *[f'{index}_gap' for index in indices]]
    return pd.DataFrame(list(rows.values()), columns=columns)



##### Tools for latent models #####

# Avoid warnings because of deprecated functions in StepMix
warnings.filterwarnings('ignore', module='sklearn.*', category=FutureWarning)


def elbow_method(df, val_index):
    res = df.dropna(subset=[val_index])
    x = res['n_clust']
    y = res[val_index]

    if val_index == 'entropy':
        knee_locator = KneeLocator(x, y, curve='concave', direction='increasing')
    else:
        knee_locator = KneeLocator(x, y, curve='convex', direction='decreasing')
    
    return res[res["n_clust"] == knee_locator.knee]['n_clust'].iloc[0]


def lrt(models):
    l2_red = (models['LL'].iloc[0] - models['LL']) / models['LL'].iloc[0]
    lik_rat = 2 * (models['LL'] - models['LL'].iloc[0])
    d_df = models['df'] - models['df'].iloc[0]
    p_val = 1 - chi2.cdf(lik_rat, d_df)

    results = pd.DataFrame({
        'n clust': models['n_clust'],
        'L2 reduction': l2_red,
        'LR ratio': lik_rat,
        'LR pval': p_val
    }, index=models.index)
    
    return results


def blrt(null_model, alternative_model, X, Y=None, sample_weight=None, n_repetitions=30, random_state=42):
    n_samples = X.shape[0]

    # Fit both models on real data
    null_model.fit(X, Y)
    alternative_model.fit(X, Y)
    real_stat = 2 * (alternative_model.score(X, Y) - null_model.score(X, Y)) * n_samples

    # Bootstrap null model
    _, stats_null = bootstrap(
        null_model,
        X,
        Y,
        n_repetitions=n_repetitions,
        identify_classes=False,
        sampler=null_model,
        random_state=random_state,
        parametric=False,
        progress_bar=False,
    )
    _, stats_alternative = bootstrap(
        alternative_model,
        X,
        Y,
        n_repetitions=n_repetitions,
        identify_classes=False,
        sampler=null_model,
        random_state=random_state,
        parametric=False,
        progress_bar=True,
    )
    gen_stats = 2 * (stats_alternative["LL"] - stats_null["LL"])
    b = np.sum(gen_stats > real_stat)

    return b / n_repetitions


def _blrt_worker(k, model, X, Y, sample_weight, n_repetitions, random_state):
    null_model = clone(model)
    null_model.set_params(n_components=k)
    alternative_model = clone(model)
    alternative_model.set_params(n_components=k + 1)

    p = blrt(
        null_model,
        alternative_model,
        X,
        Y=Y,
        sample_weight=sample_weight,
        n_repetitions=n_repetitions,
        random_state=random_state)

    return (f"{k} vs. {k + 1} classes", p)


def blrt_sweep_custom(
    model, X, Y=None, sample_weight=None, low=1, high=5, n_repetitions=30,
    random_state=42, verbose=False, n_jobs=1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_blrt_worker)(
            k, model, X, Y, sample_weight, n_repetitions, random_state)
        # for k in range(low, high))
        for k in reversed(range(low, high)))

    test_string, p_values = zip(*results)
    df = pd.DataFrame({"Test": test_string, "p": p_values}).set_index("Test")

    if verbose:
        print("\nBLRT Sweep Results")
        print(df.round(4))
    return df
