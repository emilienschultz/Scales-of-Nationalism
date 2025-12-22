import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA



# Plot datapoints and clusters
def plot_clusters(data, pred_clust, title, filename):
    unique_clusters, counts = np.unique(pred_clust, return_counts=True) # Identify clusters
    cluster_sizes = dict(zip(unique_clusters, counts)) # Map their label to their size
    
    # Create a 2D space with PCA
    pca = PCA(n_components=2)
    reduced_space = pca.fit_transform(data)
    explained_var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8,6))

    # Collect all hull vertices for clusters
    hull_vertices = []
    hull_colors = []
    for i in unique_clusters:
        if i >= 0: # Ignore noise
            cluster_points = reduced_space[pred_clust == i]
            if len(cluster_points) > 2:
                hull = ConvexHull(cluster_points)
                hull_vertices.append((
                    cluster_points[hull.vertices, 0],
                    cluster_points[hull.vertices, 1]
                ))
                hull_colors.append(i)

    # Separate clusters and noise
    mask_noise = pred_clust == -1
    mask_clusters = pred_clust >= 0

    # Plot noise points with '*' if any
    if np.any(mask_noise):
        plt.scatter(reduced_space[mask_noise, 0], reduced_space[mask_noise, 1], 
                    c='gray', marker='*', s=50, label="Noise")
    
    # Plot clusters
    scatter = plt.scatter(reduced_space[mask_clusters, 0], reduced_space[mask_clusters, 1], 
                          c=pred_clust[mask_clusters], cmap='tab10', 
                          s=15, edgecolors='k', label="Clusters")

    # Draw convex hulls for clusters
    for vertices, i in zip(hull_vertices, hull_colors):
        plt.fill(vertices[0], vertices[1], 
                 alpha=0.3,
                 color=scatter.cmap(scatter.norm(i)))

    # Modify legend to include cluster sizes
    handles, labels = scatter.legend_elements()
    new_labels = []
    for l in labels:
        try:
            cluster_id = int(l.strip('$\\mathdefault{}')) # Extract numeric part
            new_labels.append(f"Clst. {cluster_id+1} | n={cluster_sizes[cluster_id]}")
        except ValueError:
            new_labels.append(l) # Fallback if parsing fails

    # Add noise manually to legend if it exists
    if np.any(mask_noise):
        handles.append(plt.Line2D([0], [0], marker='*', color='gray', linestyle='None', markersize=10))
        new_labels.append("Noise")

    ax.axhline(y=0, color='grey', linestyle='dashed', linewidth=1)
    ax.axvline(x=0, color='grey', linestyle='dashed', linewidth=1)
    plt.legend(handles, new_labels, title="")
    plt.xlabel(f"First Dim. ({explained_var[0]:.2f}%)")
    plt.ylabel(f"Second Dim. ({explained_var[1]:.2f}%)")
    # plt.title(title)

    plt.savefig(f'output/plots/{filename}.png', format='png')
    plt.show()


# Plot response patterns
def plot_cluster_profiles(
    features,
    cluster_labels,
    feature_names,
    sd,
    title,
    filename,
    class_names = None,
    alpha=0.4):
    """
    Create a profile plot for clustering results, supporting both probabilistic
    (e.g., LCA, GMM) and deterministic (e.g., k-means) clustering methods.
    
    Parameters:
    -----------
    features : array-like or pandas.DataFrame
        The original feature matrix used for clustering (n_samples, n_features)
    cluster_labels : array-like
        Cluster assignments for each sample (n_samples,)
    feature_names : list, optional
        Names of the features (default: None, will use indices or DataFrame columns)
    class_names : list, optional
        Names of the classes (default: None, will use indices)
    sd : float
        Number of standard deviations around the mean to plot
    alpha : float, optional
        Base transparency for the scatter points
    """
    # Convert features to numpy array if it's a DataFrame
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns.tolist()
        features = features.to_numpy()
    
    # Convert cluster_labels to numpy array if it's a Series
    if isinstance(cluster_labels, pd.Series):
        cluster_labels = cluster_labels.to_numpy()
    
    # Handle NaN values
    features = np.nan_to_num(features, nan=np.nanmean(features))
    
    n_features = features.shape[1]
    n_classes = len(np.unique(cluster_labels))
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    if class_names is None:
        class_names = [f'Clst. {i+1}' for i in range(n_classes)]
        
    # Create figure
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Calculate class centroids and confidence intervals
    centroids = []
    std_devs = []
    cluster_sizes = []  # New list to store cluster sizes
    
    for class_idx in range(n_classes):
        class_mask = cluster_labels == class_idx
        class_data = features[class_mask]
        cluster_sizes.append(len(class_data))  # Store cluster size
        
        if len(class_data) > 0:
            # Calculate centroid
            centroid = np.nanmean(class_data, axis=0)
            centroids.append(centroid)
            
            # Calculate standard deviations
            std_dev = np.nanstd(class_data, axis=0)
            std_devs.append(std_dev)
            
        else:
            # Handle empty classes
            centroids.append(np.zeros(n_features))
            std_devs.append(np.zeros(n_features))
    
    # Convert to numpy arrays for vectorized operations
    centroids = np.array(centroids)
    std_devs = np.array(std_devs)
    
    # Plot for each class
    x = np.arange(n_features)
    width = 0.8 / n_classes
    
    for i in range(n_classes):
        # Offset x positions for each class
        x_pos = x - (width * (n_classes-1)/2) + (i * width)
        
        # Plot standard deviation boxes
        for j in range(n_features):
            # Clamp values to the values range
            lower = max(1, centroids[i][j] - std_devs[i][j]*sd/2)
            upper = min(5, centroids[i][j] + std_devs[i][j]*sd/2)
            height = upper - lower

            rect = plt.Rectangle((x_pos[j] - width/2, lower),
                        width, height,
                        alpha=0.2, color=f'C{i}')
            ax.add_patch(rect)   
        
        # Plot centroids with updated legend label including cluster size
        ax.scatter(x_pos, centroids[i], color=f'C{i}', 
                label=f'{class_names[i]} | n={cluster_sizes[i]}', 
                marker='*', zorder=5)
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylim(0.9, 5.1)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel('Survey Answers')
    ax.legend(title='')
    ax.grid(True, axis='y', alpha=0.3)
    # ax.set_title(f"Answer Patterns (mean ± {sd} sd) within the Best Partition According to the {title} Index")
    plt.tight_layout()

    plt.savefig(f'output/plots/{filename}_patterns.png', format='png')
    plt.show()
