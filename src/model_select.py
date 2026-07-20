import pandas as pd
import numpy as np
import warnings

from joblib import Parallel, delayed
from kneed import KneeLocator
from scipy.stats import chi2, chi2_contingency
from sklearn.base import clone
from src.model_fit import build_latent_model, do_StepMix, do_kmeans, do_AHC, do_hdbscan
from stepmix.stepmix import bootstrap



##### Gap statistic #####

def bootstrap_gap(data, controls, n, model, params, iter_num):
    # Create a random dataset. For categorical latent models the columns are
    # integer categories min..max, so drawing on [min, max + 1) yields a
    # discrete uniform over the observed categories once truncated to int;
    # for continuous data (kmeans / AHC / continuous latent) the reference
    # box must stop at the observed max
    if model == 'latent' and 'categorical' in str(params.get('msrt', '')):
        high = data.max(axis=0) + 1
    else:
        high = data.max(axis=0)

    rand_data = np.random.uniform(
        low=data.min(axis=0),
        high=high,
        size=data.shape)
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
    
    # Add iteration number
    res = pd.DataFrame([res])
    res['bootstrap_iter'] = iter_num + 1
    
    return res


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
            
            # Calculate gap statistic and s value, oriented so that a larger
            # gap always means the model beats the random reference:
            # Davies-Bouldin is lower-is-better (Tibshirani's original
            # orientation); silhouette, CH and Dunn are higher-is-better
            if index == 'davies_bouldin':
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


def get_gap(gap_values, model, params, index):
    # Subset gap_values to the right model and params
    rows_id = ((gap_values['model'] == model) & (gap_values['params'] == params))
    df = gap_values[rows_id].reset_index(drop=True)

    # Extract gap and s values
    gap = df[f'{index}_gs']
    s = df[f'{index}_s']

    # Select rows such that GS(k) >= GS(k+1) - s(k+1)
    # Skipping the last row and adjusting for index-based calculations
    n_min = df['n_clust'].min()
    stats = []
    
    for i in range(0, len(df) - 1):
        stat = gap[i] - gap[i+1] + s[i+1]
        if stat >= 0: 
            stats.append([i+n_min, stat])

    # Return optimal cluster number
    stats = np.array(stats)
    if stats.size == 0:
        best_n = 'none'
    else:
        best_n = int(stats[np.argmin(stats[:, 1]), 0])

    return best_n



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
