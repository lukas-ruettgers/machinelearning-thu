import matplotlib.pyplot as plt # Graph plotting
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import idx2numpy
from pca import pca
import time
import umap
 

# Constants
PLOT_SHOW = False
PLOT_SAVE = True
WRITE_TO_FILE = True
DIGIT = 3
PC_CUTOFF = 45
ZERO_TOL = 1e-16   

EXECUTE_UNIFORM_TEST = False
EXECUTE_TASK1 = True
EXECUTE_TASK2 = True
EXECUTE_TASK3 = True

TASK1_EXPERIMENT_NAME = "Spectral Clustering on entire MNIST digits dataset"
TASK2_EXPERIMENT_NAME = f"Spectral Clustering on first {PC_CUTOFF} PCAs of MNIST digits dataset"
TASK3_EXPERIMENT_NAME = f"Spectral Clustering on first {PC_CUTOFF} PCAs of MNIST digit {DIGIT}"

TASK1_FILE_PREFIX = "SPECTRAL_UMAP"
TASK2_FILE_PREFIX = "SPECTRAL_PCA_Embedding_UMAP"
TASK3_FILE_PREFIX = f"DIGIT{DIGIT}_SPECTRAL_PCA_Embedding_UMAP"
TASK3_FILE_PREFIX_REPRESENTATIVES = f"DIGIT{DIGIT}_REPR"

# UMAP Hyperparamters from last experiment
N_NEIGHBORS = 15 # default: 15
MIN_DIST = 0.1 # default: 0.1
SPREAD = 1.0 # default: 1.0
METRIC = 'minkowski' # ['euclidian' (default), 'mahalanobis', 'minkowski']
MINKOWSKI_P = 2

# Sampling for feasible representative calculations
SAMPLE_SIZE = 1000

# K-Means hyperparameters
K_MEANS_MAX_ITERATIONS = 300
K_MEANS_TOL = 1e-5
K = 10
LOWER_K = 5 # inclusive
UPPER_K = 15 # exclusive
TASK3_LOWERK = 2
TASK3_UPPERK = 5

# Visualization
color_idxs = [
    mcolors.CSS4_COLORS['indianred'],
    mcolors.CSS4_COLORS['orange'],
    mcolors.CSS4_COLORS['goldenrod'],
    mcolors.CSS4_COLORS['palegreen'],
    mcolors.CSS4_COLORS['seagreen'],
    mcolors.CSS4_COLORS['turquoise'],
    mcolors.CSS4_COLORS['skyblue'],
    mcolors.CSS4_COLORS['steelblue'],
    mcolors.CSS4_COLORS['slateblue'],
    mcolors.CSS4_COLORS['darkorchid'],
    mcolors.CSS4_COLORS['mediumvioletred'],
    mcolors.CSS4_COLORS['deeppink'],
    mcolors.CSS4_COLORS['gold'],
    mcolors.CSS4_COLORS['greenyellow'],
    mcolors.CSS4_COLORS['navy']
]
color_gray = mcolors.CSS4_COLORS['gray']
legend_patches = [mpatches.Patch(color = color_idx, label = f"{i}") for i, color_idx in enumerate(color_idxs)]
POINT_SIZE = plt.rcParams['lines.markersize'] / 16

def principal_component_analysis(data, var_percentage = 0.95):
    model = pca(n_components=var_percentage)
    results = model.fit_transform(data)
    return model, results   

def standard_deviation(data):
    """Compute the standard deviation of a multi-feature dataset."""
    n_images = data.shape[0]
    mean = np.sum(data, axis=0) / n_images
    squared_mean = np.sum(np.square(data), axis=0) / n_images
    var = squared_mean - np.square(mean)

    # Avoid negative values
    var[var<0] = 0
    std_dev = np.sqrt(var)
    return std_dev

def distance(x,y):
    dist = np.sum(np.square(x-y))
    dist = max(0, dist)
    return np.sqrt(dist)

def k_means_clustering(data, k=K, iterations=K_MEANS_MAX_ITERATIONS):
    """
    Use Lloyd algorithm to compute optimal means of k clusters

        data: array((N,d))
    """
    
    # Initialize means
    n_data = data.shape[0]
    n_dim = data.shape[1]

    """ 
    Alternative method: Choose next mean as the point with the largest average distance to the prior means.
    Since there were some extreme outliers in the dataset, this approach yielded unsatisfactory results.
    first_mean_idx = np.random.randint(0, n_data)
    means[0] = data[first_mean_idx]
    mean_idxs = [first_mean_idx]
    
    avg_dist_sum = 0.0
    avg_dist = np.zeros(n_data)
    for i in range(1, k):
        # Compute the average distance of each point to cluster means
        avg_dist_sum = 0.0
        for i_x in range(n_data):
            avg_dist[i_x] = 0.0
            for j in range(i):
                avg_dist[i_x] += distance(means[j], data[i_x])
            avg_dist[i_x] /= i

            # Don't select one mean twice
            if i_x in mean_idxs:
                avg_dist[i_x] = 0.0
            avg_dist_sum += avg_dist[i_x]

        # Choose vector as mean with a probability proportional to the average squared distance to the prior means
        r = np.random.rand()
        next_mean_idx = i
        ratio = 0.0
        for i_x in range(n_data):
            ratio += avg_dist[i_x]/avg_dist_sum
            if ratio > r:
                next_mean_idx = i_x
                break
        
        means[i] = data[next_mean_idx]
        mean_idxs.append(next_mean_idx)
    """

    # Randomly initialize means. Ensure that there are no duplicates.
    mean = np.sum(data, axis=0) / n_data
    squared_mean = np.sum(np.square(data), axis=0) / n_data
    var = squared_mean - np.square(mean)
    std_dev = np.sqrt(var)
    means = np.zeros((k,n_dim))
    mean_idxs = np.zeros(k,dtype=int)
    for i in range(k):
        while True:
            next_idx = np.random.randint(0, n_data)
            if next_idx not in mean_idxs:
                mean_idxs[i] = next_idx
                means[i] = data[next_idx]
                break
    
    # Prevent outliers
    means = (means + mean)/2

    # Iteratively reassign points and update means
    cluster = np.zeros(n_data,dtype=int)
    new_means = np.zeros((k,n_dim))
    new_cluster_sizes = np.zeros(k, dtype=int)
    min_total_distance = -1.0
    r = 0
    for it in range(2*iterations):
        new_means.fill(0)
        new_cluster_sizes.fill(0)
        # Reassign points
        for i in range(n_data):
            min_mean = 0
            min_dist = distance(means[0], data[i])
            # if it == 0 and i% 10 == 0:
            #     print(f"{i}:\t{data[i]}")
            #     print(f"Mean {0}:\t{means[0]}, dist:\t{min_dist}")
            for j in range(1,k):
                dist = distance(means[j], data[i])
                # if it == 0 and i% 10 == 0:
                #     print(f"Mean {j}:\t{means[j]}, dist:\t{dist}")
            
                if dist < min_dist:
                    min_mean = j
                    min_dist = dist
            # if it == 0 and i% 10 == 0:
            #     print(f"")

            cluster[i] = min_mean
            new_cluster_sizes[min_mean] += 1
            new_means[min_mean] += data[i]

        # if it <= 3:
        #     print(f"It {it}, cluster sizes: {new_cluster_sizes}")
        # Update means
        for i in range(k):
            # Scale mean if cluster not empty
            if new_cluster_sizes[i] > 0:
                new_means[i] /= new_cluster_sizes[i]
                continue
            
            # Edge case: empty clusters.
            # Randomly choose other point as new cluster mean.
            while(new_cluster_sizes[i]==0):
                r = np.random.randint(0,n_data)
                if new_cluster_sizes[cluster[r]] < 2:
                    continue
                new_cluster_sizes[cluster[r]] -= 1
                new_means[cluster[r]] -= data[r]
                
                new_cluster_sizes[i] = 1
                new_means[i] = data[r]
                cluster[r] = i

        # Stop if means converge
        total_distance = 0.0
        for mean, new_mean in zip(means, new_means):
            total_distance += distance(mean, new_mean)
        
        # Even if maximum number of iterations exceeded, ensure that result is near the so far optimal
        if total_distance < min_total_distance or min_total_distance < 0:
            min_total_distance = total_distance

        if total_distance < K_MEANS_TOL or it > iterations and total_distance < 2*min_total_distance:
            break

        means = np.copy(new_means)
    
    return means, cluster

def affinity_matrix(images_flattened, std_dev):
    """Compute the affinity matrix."""
    n_images = images_flattened.shape[0]
    n_dim = images_flattened.shape[1]
    affinity_matrix = np.zeros((n_images, n_images))
    row_sum = np.zeros((n_images,1))
    
    # Avoid divison by zero
    std_dev[std_dev <= ZERO_TOL] = ZERO_TOL

    # First, obtain the unnormalized affinity matrix
    for i in range(n_images):
        # Ignore dimensions with very small values 
        x = np.copy(images_flattened[i])
        x[std_dev<=ZERO_TOL] = 0
        for j in range(i+1, n_images):
            y = np.copy(images_flattened[j])
            y[std_dev<=ZERO_TOL] = 0
            dist_sum = np.sum(np.square((x-y)/std_dev))
            # Scale distances because they tend to be large in such high dimensions
            dist_sum /= np.sqrt(n_dim)
            dist = np.exp((-0.5)*dist_sum)
            affinity_matrix[i][j] = dist
            affinity_matrix[j][i] = dist

            # Compute sum of rows on-the-fly
            row_sum[i] += dist
            row_sum[j] += dist
    
    # Normalize each element by left and right multiplication of degree matrix
    row_sum[row_sum < 0] = 0
    row_sum = np.sqrt(row_sum)
    degree_scale = row_sum * np.transpose(row_sum)
    
    # Handle possible divison by zero.
    affinity_matrix[degree_scale<=0] = 0
    degree_scale[degree_scale<=0] = 1    
    affinity_matrix /= degree_scale
    return affinity_matrix

def spectral_clustering(images_flattened, k=K):
    """Implementation of the NJW algorithm to perform a spectral clustering into k clusters."""

    # Since the entire dataset is too large, sample a representative fraction to proceed with.
    n_images = images_flattened.shape[0]
    sample_idxs = [np.random.randint(0,n_images) for _ in range(SAMPLE_SIZE)]
    samples = np.take(images_flattened, sample_idxs, axis=0)
    std_deviation = standard_deviation(samples)
    aff_mat = affinity_matrix(samples, std_deviation)

    # Obtain first k eigenvectors and normalize them along rows
    eigenvals, eigenvecs = np.linalg.eig(aff_mat)
    embed_mat = eigenvecs.real[:,:k]
    embed_mat = np.array(embed_mat)
    row_sums = np.sum(np.square(embed_mat), axis=1, keepdims=True)
    row_sums[row_sums <= 0] = ZERO_TOL
    row_sums = np.sqrt(row_sums)
    embed_mat /= row_sums

    means, cluster_samples = k_means_clustering(embed_mat, k)

    # For each mean in the embedding determine the sample with the closest embedding
    min_dist = np.full(k,-1)
    representative_sample_idxs = np.full(k,-1)
    for i in range(SAMPLE_SIZE):
        cluster_i = cluster_samples[i]
        dist = distance(embed_mat[i], means[cluster_i])
        if dist < min_dist[cluster_i] or min_dist[cluster_i] < 0:
            min_dist[cluster_i] = dist
            representative_sample_idxs[cluster_i] = i

    # Map all images not in the samples to the cluster with the closest representative
    cluster_total = np.full(n_images, -1)
    cluster_total.put(sample_idxs,cluster_samples)
    for i in range(n_images):
        if cluster_total[i] >= 0:
            continue
        min_dist = distance(images_flattened[i], images_flattened[sample_idxs[representative_sample_idxs[0]]])
        cluster_total[i] = 0
        for j in range(1, k):
            dist = distance(images_flattened[i], images_flattened[sample_idxs[representative_sample_idxs[j]]])
            if dist < min_dist:
                min_dist = dist
                cluster_total[i] = j

    return representative_sample_idxs, sample_idxs, cluster_total

def score(means_idxs, cluster_idx, data):
    """ 
    Compute a metric in BIC fashion that balances the model expressivity (k) and the clustering likelihood.
    Here, the clustering likelihood is approximated by sqrt(1- intra_cluster_distance/inter_cluster_distance). 
    """
    n_data = data.shape[0]
    k = means_idxs.shape[0]
    
    # Compute the average distance from a point from its cluster mean.
    avg_intra_cluster_dist = 0.0
    for i in range(n_data):
        i_mean = means_idxs[cluster_idx[i]]
        avg_intra_cluster_dist += distance(data[i],data[i_mean])
    avg_intra_cluster_dist = avg_intra_cluster_dist/n_data

    # Compute the average distance between cluster means.
    avg_inter_cluster_dist = 0.0
    for i in range(k):
        for j in range(i+1, k):
            avg_inter_cluster_dist += distance(data[means_idxs[i]], data[means_idxs[j]])
    avg_inter_cluster_dist = avg_inter_cluster_dist/(k*(k-1)/2)

    # The smaller the ratio between intra cluster distance and inter cluster distance, the better.
    separability_ratio = avg_intra_cluster_dist/avg_inter_cluster_dist
    total_score = 2*np.log(separability_ratio) + np.log(np.log(n_data)) * k
    return total_score


def task1(images_flattened, lower_k = LOWER_K, upper_k = UPPER_K, experiment_name = TASK1_EXPERIMENT_NAME, file_prefix="SPECTRAL_UMAP", log_results = False, point_size=POINT_SIZE/40):
    """
    Perform Spectral Clustering on an images dataset and visualize the clustering with a UMAP embedding.
    lower_k and upper_k define the boundaries in which to try out each possible number of clusters k.
    If log_results = True, return the model and score results for each clustering, else return empty arrays.
    """
    umap_model = umap.UMAP(
        n_neighbors = N_NEIGHBORS,
        min_dist = MIN_DIST,
        metric = METRIC,
        metric_kwds = {"p": MINKOWSKI_P},
        spread = SPREAD
    )
    embedding = umap_model.fit_transform(images_flattened)
    umap_embedding = embedding if log_results else []
    with open("results.txt", "a") as res_file:
        if WRITE_TO_FILE:
            res_file.write(f"\r\nExperiment: {experiment_name}. Time: {time.strftime('%a %d %b %Y, %I:%M%p')}\r\n")
        means_results = []
        cluster_results = []
        score_results = []
        n_images = images_flattened.shape[0]
        for k in range(lower_k, upper_k):
            mean_idxs, sample_idxs, cluster = spectral_clustering(images_flattened, k=k)
            
            # Score the results considering both only the samples and all images
            cluster_samples = np.take(cluster, sample_idxs, axis=0)
            images_samples = np.take(images_flattened, sample_idxs, axis=0)
            bic_samples = score(mean_idxs, cluster_samples, images_samples)
            bic = score(np.take(sample_idxs,mean_idxs,axis=0), cluster, images_flattened) # Mean_idxs are still in terms of sample indexes
            if log_results:
                means_results.append(np.take(sample_idxs,mean_idxs,axis=0))
                cluster_results.append(cluster)
                score_results.append(bic)
            # print(f"Clusters: \t{k}, BIC: \t{bic}")
            if WRITE_TO_FILE:
                res_file.write(f"Clusters: \t{k}, BIC: \t{bic}, BIC Samples Only: \t{bic_samples}\r\n")

            # Visualize total clusters on UMAP embedding
            colors = [color_idxs[label] for label in cluster]
            patches = [mpatches.Patch(color = color_idxs[i], label = f"{i}") for i in range(k)]
            plt.scatter(embedding[:,0], embedding[:,1], c=colors, s=point_size)
            plt.title(f"UMAP Embedding of the MNIST Digits Spectral Clustering ($k={k}$)")
            plt.legend(handles=patches)
            if PLOT_SHOW:
                plt.show()
            if PLOT_SAVE:
                plt.savefig(f"Latex Report/Figures/{file_prefix}_{k}.png")
            plt.clf()

            # Now visualize them for only the samples
            colors = np.take(colors,sample_idxs,axis=0)
            plt.scatter(np.take(embedding,sample_idxs,axis=0)[:,0], np.take(embedding,sample_idxs,axis=0)[:,1], c=colors, s=point_size*n_images/SAMPLE_SIZE*5)
            plt.title(f"UMAP of the Spectral Clustering ($k={k}$) - SAMPLES ONLY")
            plt.legend(handles=patches)
            if PLOT_SHOW:
                plt.show()
            if PLOT_SAVE:
                plt.savefig(f"Latex Report/Figures/{file_prefix}_SAMPLESONLY_{k}.png")
            plt.clf()

        return umap_embedding, means_results, cluster_results, score_results 

def task2(images_flattened, pc_size=PC_CUTOFF, lower_k = LOWER_K, upper_k = UPPER_K, experiment_name=TASK2_EXPERIMENT_NAME, file_prefix=TASK2_FILE_PREFIX, log_results = False, point_size=POINT_SIZE/40):
    """Perform Spectral Clustering on the first few PCAs of the images."""
    # Obtain PCAs
    model, results = principal_component_analysis(images_flattened, var_percentage=0.85)
    pca_embedding = np.array([getattr(results['PC'],f"PC{i+1}") for i in range(pc_size)])
    pca_embedding = np.transpose(pca_embedding) # First dimension: Images, second dimension: Principal Components
    pca_results = pca_embedding if log_results else []
    umap_embedding, means_results, cluster_results, score_results = task1(pca_embedding, 
                                                          lower_k = lower_k,
                                                          upper_k = upper_k,
                                                          experiment_name=experiment_name, 
                                                          file_prefix=file_prefix, 
                                                          log_results=log_results,
                                                          point_size=point_size
                                                          )
    return pca_results, umap_embedding, means_results, cluster_results, score_results
    
def task3(images, images_flattened, labels, pc_size=PC_CUTOFF):    
    # First compute PCs with flattened images
    subclass_images = []
    subclass_images_idx = []
    for i, image in enumerate(images_flattened):
        if labels[i] == DIGIT:
            subclass_images.append(image)
            subclass_images_idx.append(i)
    n_images = len(subclass_images)
    
    pca_embedding, umap_embedding, means_results, cluster_results, score_results = task2(subclass_images, 
                                                          pc_size = pc_size,
                                                          lower_k = TASK3_LOWERK, 
                                                          upper_k = TASK3_UPPERK,
                                                          experiment_name=TASK3_EXPERIMENT_NAME, 
                                                          file_prefix=TASK3_FILE_PREFIX, 
                                                          log_results=True,
                                                          point_size=POINT_SIZE
                                                          )
    
    # Decide best k
    min_score = score_results[0]
    k_opt = 0
    for i, score in enumerate(score_results):
        if score < min_score:
            min_score = score
            k_opt = i
    
    cluster_opt = cluster_results[k_opt]
    mean_idxs_opt = means_results[k_opt]
    colors = [color_idxs[label] for label in cluster_opt]
    
    # Increase size of representatives to make them more visible
    # Simultaneously save their images
    point_sizes = [POINT_SIZE]*n_images
    if PLOT_SAVE:
        for i, idx in enumerate(mean_idxs_opt):
            point_sizes[idx] = plt.rcParams['lines.markersize']**2    
            plt.imsave(f"Latex Report/Figures/{TASK3_FILE_PREFIX_REPRESENTATIVES}{i}_{int(time.time())}.png", images[subclass_images_idx[idx]], 
                    cmap='gray', vmin=0, vmax=255)

        plt.scatter(umap_embedding[:,0], umap_embedding[:,1], c=colors, s=point_sizes)
        plt.title(f"UMAP Embedding of clustered MNIST Digit {DIGIT} Images")
        patches = [mpatches.Patch(color = color_idxs[i], label = f"{i}") for i in range(k_opt)]
        plt.legend(handles=patches)
        plt.savefig(f"Latex Report/Figures/{TASK3_FILE_PREFIX}_KOPT{k_opt}_{int(time.time())}.png")
    

def test_with_uniform_data():
    test_data = []
    for _ in range(3*SAMPLE_SIZE):
        r = np.random.randint(0,3)
        if r == 0:
            test_data.append([np.random.rand()-0.5,np.random.rand()-0.5])
        elif r == 1:
            test_data.append([np.random.rand()+4.5,np.random.rand()-0.5])
        else:
            test_data.append([np.random.rand()-0.5,np.random.rand()+4.5])
    task1(np.array(test_data), 
          experiment_name="Uniform data from 3 clusters", 
          file_prefix="UNIFORM3",
          lower_k=2,
          upper_k=5,
          point_size=POINT_SIZE
          )
    

def main():
    if not PLOT_SHOW:
        print("No figures will be shown. Set PLOT_SHOW to True to enable this function.")
    
    if not PLOT_SAVE:
        print("No figures will be saved to the Figures folder. Set PLOT_SAVE to True to enable this function.")

    if not WRITE_TO_FILE:
        print("No result scores will be written to the results file. Set WRITE_TO_FILE to True to enable this function.")

    images = idx2numpy.convert_from_file("data2forEx8+/train-images.idx3-ubyte")
    labels = idx2numpy.convert_from_file("data2forEx8+/train-labels.idx1-ubyte")

    N_IMAGES = images.shape[0]
    images_flattened = np.reshape(images, (N_IMAGES,-1))

    ### TEST
    if EXECUTE_UNIFORM_TEST:
        test_with_uniform_data()

    ### TASK 1
    if EXECUTE_TASK1:
        task1(images_flattened, experiment_name=TASK1_EXPERIMENT_NAME, file_prefix=TASK1_FILE_PREFIX)
    else:
        print("The code for task 1 will not be executed. Set EXECUTE_TASK1 to True to enable this function.")
    
    ### TASK 2
    if EXECUTE_TASK2:
        task2(images_flattened, pc_size=PC_CUTOFF, experiment_name=TASK2_EXPERIMENT_NAME, file_prefix=TASK2_FILE_PREFIX)
    else:
        print("The code for task 2 will not be executed. Set EXECUTE_TASK2 to True to enable this function.")

    
    ### TASK 3
    if EXECUTE_TASK3:
        task3(images, images_flattened, labels, pc_size=PC_CUTOFF)
    else:
        print("The code for task 3 will not be executed. Set EXECUTE_TASK3 to True to enable this function.")
    
    return 0
    

if __name__=="__main__":
    main()