import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# Avoid throwing errors when a CVI is undefined
def sil_score(data, pred_clust):
    try: 
        sil_score = silhouette_score(data, pred_clust, metric='manhattan')
    except ValueError:
        sil_score = np.nan
    return sil_score

def ch_score(data, pred_clust):
    try:
        ch_score = calinski_harabasz_score(data, pred_clust)
    except ValueError:
        ch_score = np.nan
    return ch_score

def db_score(data, pred_clust):
    try:
        db_score = davies_bouldin_score(data, pred_clust)
    except ValueError:
        db_score = np.nan
    return db_score


# Function to compute the Dunn score 43
def dunn_score(data, pred_clust, metric='cityblock'):
    data = np.asarray(data)
    pred_clust = np.asarray(pred_clust)
    clusters = np.unique(pred_clust)
    
    # Compute min centroid distance (for separation)
    min_centroid_dist = None
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            # Get data points for each cluster
            cluster_i = data[pred_clust == clusters[i]]
            cluster_j = data[pred_clust == clusters[j]]
            
            # Compute centroids
            centroid_i = np.mean(cluster_i, axis=0)
            centroid_j = np.mean(cluster_j, axis=0)
            
            # Reshape centroids for cdist (needs 2D arrays)
            centroid_i = centroid_i.reshape(1, -1)
            centroid_j = centroid_j.reshape(1, -1)
            
            # Compute distance between centroids using specified metric
            centroid_dist = cdist(centroid_i, centroid_j, metric=metric)[0, 0]
            
            if min_centroid_dist is None or centroid_dist < min_centroid_dist:
                min_centroid_dist = centroid_dist
    
    # Compute max cluster diameter (for cohesion)
    max_diameter = 0
    for cluster in clusters:
        clust = data[pred_clust == cluster]
        if len(clust) > 0:
            centroid = np.mean(clust, axis=0)
            centroid_2d = centroid.reshape(1, -1)
            # Use same distance metric for consistency
            diam = np.mean(cdist(clust, centroid_2d, metric=metric))
            max_diameter = max(max_diameter, diam)        
    
    # Compute Dunn score
    if max_diameter > 0 and min_centroid_dist is not None:
        dunn_score = min_centroid_dist / max_diameter
    else:
        dunn_score = np.nan
    
    return dunn_score


# Clusters min an max sizes
def clust_size(pred_clust):
    cluster_sizes = Counter(pred_clust)
    min_size = min(cluster_sizes.values())
    max_size = max(cluster_sizes.values())
    
    return min_size, max_size


# Return all model parameters and CVI at once
def get_metrics(model, params, n, data, pred_clust, **additional_metrics):

    # Remove noise (HDBSCAN -1 labels): CVIs and cluster sizes are computed
    # on the clustered points only, so that scores stay comparable with
    # models that assign every point to a cluster
    noise = pred_clust == -1
    denoised_data = data[~noise]
    denoised_pred_clust = pred_clust[~noise]

    if len(denoised_pred_clust) > 0:
        min_size, max_size = clust_size(denoised_pred_clust)
    else:
        min_size, max_size = 0, 0

    base_metrics = {
        'model': model,
        'params': params,
        'n_clust': n,
        'min_clust_size': min_size,
        'max_clust_size': max_size,
        'silhouette': float(sil_score(denoised_data, denoised_pred_clust)),
        'calinski_harabasz': float(ch_score(denoised_data, denoised_pred_clust)),
        'davies_bouldin': float(db_score(denoised_data, denoised_pred_clust)),
        'dunn': float(dunn_score(denoised_data, denoised_pred_clust))
    }

    base_metrics.update(additional_metrics)
    return base_metrics
