import pandas as pd
import numpy as np
import warnings

from scipy.spatial.distance import cdist, mahalanobis
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from stepmix.stepmix import StepMix

from src.model_eval import get_metrics



##### Latent models #####

# Avoid warnings because of deprecated functions in StepMix
warnings.filterwarnings('ignore', module='sklearn.*', category=FutureWarning)

def build_latent_model(n, msrt, covar):
    opt_params = {
        'method': 'gradient',
        'intercept': True,
        'max_iter': 500}
    
    if covar == 'without':
        latent_mod = StepMix(
            n_components = n,
            measurement = msrt,
            n_init = 7 if 'nan' in msrt else 3,
            abs_tol=1e-4,
            rel_tol=1e-4,
            init_params = 'random' if 'nan' in msrt else 'kmeans',
            structural_params = opt_params,
            progress_bar = 0)
        
    elif covar == 'with':
        latent_mod = StepMix(
            n_components = n,
            measurement = msrt,
            n_init = 7 if 'nan' in msrt else 3,
            abs_tol=1e-4,
            rel_tol=1e-4,
            init_params = 'random' if 'nan' in msrt else 'kmeans',
            structural = 'covariate',
            structural_params = opt_params,
            progress_bar = 0)

    return latent_mod


def do_StepMix(data, controls, n, msrt, covar, weights = None, refit = False):
    # StepMix strictly type-checks n_components against Python int, so a
    # numpy.int64 (e.g. coming from a DataFrame column) is rejected with a
    # misleading "value not recognized" error. Coerce to a plain int.
    n = int(n)
    latent_mod = build_latent_model(n, msrt, covar)
    latent_mod.fit(data, controls, weights)
    pred_clust = latent_mod.predict(data, controls)

    if refit: 
        return pred_clust

    else:
        unique_patterns = data.drop_duplicates().shape[0]
        mod_params = latent_mod.n_parameters
        df = unique_patterns - mod_params
        
        # Extract model coefficients
        coeffs = latent_mod.get_parameters_df()
        coeffs = coeffs.reset_index()
        coeffs = coeffs[['class_no', 'variable', 'value']]
        # Extract posterior and modal probabilities
        post_probs = latent_mod.predict_proba(data, controls)
        mod_probs = np.max(post_probs, axis=1)
        # Compute classification error
        classif_error =  (1 - mod_probs).sum() / len(data)
    
        return get_metrics(
            model = 'latent',
            params = {
		'msrt': msrt,
		'NAs': 'with' if 'nan' in msrt else 'without', 
		'covar': covar,
		'wgt': 'without' if weights is None else 'with'},
            n = n,
            data = data,
            pred_clust = pred_clust,
            aic = latent_mod.aic(data, controls),
            bic = latent_mod.bic(data, controls),
            sabic = latent_mod.sabic(data, controls),
            relative_entropy = latent_mod.relative_entropy(data, controls),
            classif_error = classif_error,
            df = df,
            LL = latent_mod.score(data, controls))



##### k-means #####

class FlexibleKMeans:
    """
    K-Means implementation supporting different distance metrics and center computation methods.
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters
    metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', 'chebyshev'
    center_method : str, default='mean'
        Method to compute cluster centers: 'mean', 'median', 'medoid'
    max_iter : int, default=100
        Maximum number of iterations
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.
        The final result will be the best output of n_init consecutive runs in terms of inertia.
    random_state : int or None, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_clusters, metric='euclidean', center_method='mean',
                 max_iter=100, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.center_method = center_method
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        
        # Define mapping from user-friendly names to scipy metrics
        self.metric_mapping = {
            'euclidean': 'euclidean',
            'manhattan': 'cityblock',
            'chebyshev': 'chebyshev'
        }
        
        # Validate inputs
        valid_metrics = list(self.metric_mapping.keys())
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
            
        valid_centers = ['mean', 'median', 'medoid']
        if center_method not in valid_centers:
            raise ValueError(f"center_method must be one of {valid_centers}")
            
        if self.n_init <= 0:
            raise ValueError("n_init should be > 0")
    
    def _compute_distances(self, X, centers):
        """Compute distances between points and centers using specified metric."""
        return cdist(X, centers, metric=self.metric_mapping[self.metric])
    
    def _compute_centers(self, X, labels):
        """Compute new centers using specified method."""
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            
            if len(cluster_points) == 0:
                continue
                
            if self.center_method == 'mean':
                new_centers[i] = np.mean(cluster_points, axis=0)
                
            elif self.center_method == 'median':
                new_centers[i] = np.median(cluster_points, axis=0)
                
            elif self.center_method == 'medoid':
                # For medoid, find the point that minimizes sum of distances to other points
                distances = self._compute_distances(cluster_points, cluster_points)
                medoid_idx = np.argmin(np.sum(distances, axis=1))
                new_centers[i] = cluster_points[medoid_idx]
                
        return new_centers
    
    def _single_fit(self, X, seed):
        """Perform a single run of k-means with given random seed."""
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize centers randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        centers = X[idx].copy()
        
        for iteration in range(self.max_iter):
            # Store old centers for convergence check
            old_centers = centers.copy()
            
            # Assign points to nearest center
            distances = self._compute_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            
            # Update centers
            centers = self._compute_centers(X, labels)
            
            # Check for convergence
            if np.allclose(old_centers, centers):
                n_iter = iteration + 1
                break
        else:
            n_iter = self.max_iter
            
        # Compute final inertia
        final_distances = self._compute_distances(X, centers)
        inertia = np.sum(np.min(final_distances, axis=1) ** 2)
        
        return centers, labels, inertia, n_iter
    
    def fit(self, X):
        """Fit the model to the data."""
        # Convert pandas DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X)
        
        # Initialize best solution tracking
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        best_n_iter = None
        
        # Run k-means n_init times
        for init in range(self.n_init):
            # Generate seed for this initialization
            if self.random_state is not None:
                seed = self.random_state + init
            else:
                seed = None
                
            # Perform single k-means run
            centers, labels, inertia, n_iter = self._single_fit(X, seed)
            
            # Update best solution if current one is better
            if inertia < best_inertia:
                best_centers = centers
                best_labels = labels
                best_inertia = inertia
                best_n_iter = n_iter
        
        # Store best solution
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def fit_predict(self, X):
        """Fit the model and return cluster labels."""
        return self.fit(X).labels_
    
    def predict(self, X):
        """Predict the closest cluster for each sample in X."""
        # Convert pandas DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X)
        
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    
def do_kmeans(data, n, dist, link, refit=False):
    kmeans = FlexibleKMeans(
        n_clusters = n,
        metric = dist,
        center_method = link,
        n_init = 15)
        
    pred_clust = kmeans.fit_predict(data)
    
    model = 'kmeans'
    params = {'dist': dist, 'link': link}    
    
    if refit == True: 
        return pred_clust
    else:
        return get_metrics(model, params, n, data, pred_clust)


    
##### AHC #####

def do_AHC(data, n, dist, link, refit=False):
    ahc = AgglomerativeClustering(
        n_clusters = n,
        metric = dist,
        linkage = link)
    
    ahc.fit(data)
    pred_clust = ahc.labels_

    model = 'AHC'
    params = {'dist': dist, 'link': link}

    if refit == True: 
        return pred_clust
    else:
        return get_metrics(model, params, n, data, pred_clust)



##### HDBSCAN #####

def do_hdbscan(data, dist, min_clust, min_smpl, refit=False):
    if dist == 'mahalanobis':
        cov_matrix = np.cov(data, rowvar=False)  # Compute covariance
        inv_cov_matrix = np.linalg.inv(cov_matrix)  # Compute inverse

        # Define a Mahalanobis distance function
        def mahalanobis_metric(a, b):
            return mahalanobis(a, b, inv_cov_matrix)
        dist_func = mahalanobis_metric
        
    else:
        dist_func = dist
        
    hdb = HDBSCAN(
        metric = dist_func,
        min_cluster_size = min_clust, 
        min_samples = min_smpl)
        
    pred_clust = hdb.fit_predict(data)

    model = 'HDBSCAN'
    params = {'dist': dist, 'min_clust': min_clust, 'min_smpl': min_smpl}
    n = len(set(pred_clust[pred_clust != -1]))
    noise_freq = 100 * sum(pred_clust == -1) / len(pred_clust)

    if refit == True: 
        return pred_clust
    else:
        return get_metrics(model, params, n, data, pred_clust, noise = noise_freq)
