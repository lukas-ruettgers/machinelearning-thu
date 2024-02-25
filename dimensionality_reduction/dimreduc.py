import matplotlib.pyplot as plt # Graph plotting
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import idx2numpy
from pca import pca
import time
import umap
 

# Constants
PLOT_RES = False
DIGIT = 5
PC_CUTOFF = 10
EXECUTE_TASK1 = True
EXECUTE_TASK2 = True
EXECUTE_TASK3 = True

# UMAP Hyperparamters
N_NEIGHBORS = 15 # default: 15
MIN_DIST = 0.1 # default: 0.1
SPREAD = 1.0 # default: 1.0
METRIC = 'minkowski' # ['euclidian' (default), 'mahalanobis', 'minkowski']
MINKOWSKI_P = 2

# Sampling
SAMPLE_SIZE = 1000

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
    mcolors.CSS4_COLORS['darkorchid']
]
color_gray = mcolors.CSS4_COLORS['gray']
legend_patches = [mpatches.Patch(color = color_idx, label = f"{i}") for i, color_idx in enumerate(color_idxs)]
point_size = plt.rcParams['lines.markersize'] / 16

def principal_component_analysis(data, var_percentage = 0.95):
    model = pca(n_components=var_percentage)
    results = model.fit_transform(data)
    return model, results

def plot_explained_variance(model):
    fig, ax = model.plot()
    ax.set_title("Cumulative explained variance")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Percentage of explained variance")
    return fig, ax

def biplot_principal_components(pc1, pc2, pc1_i, pc2_i, colors=None, size=plt.rcParams['lines.markersize']**2, add_legend=True, legend=legend_patches):
        fig, ax = plt.subplots()
        if colors is None:
            ax.scatter(pc1, pc2, s=size)
        else:
            ax.scatter(pc1, pc2, c=colors, s=size)

        ax.set_title(f"Biplot PC{pc1_i} vs. PC{pc2_i}")
        ax.set_xlabel(f"PC {pc1_i}")
        ax.set_ylabel(f"PC {pc2_i}")
        if add_legend:
            ax.legend(handles=legend)
        return fig, ax

def task1(images_flattened, colors):
    model, results = principal_component_analysis(images_flattened, var_percentage=0.8)

    if PLOT_RES:
        fig, ax = plot_explained_variance(model)
        fig.show()

        pc1_data = results['PC'].PC1
        pc2_data = results['PC'].PC2
        pc3_data = results['PC'].PC3
        fig, ax = biplot_principal_components(pc1_data, pc2_data, 1, 2, colors, size=point_size/16)
        fig.show()

        fig, ax = biplot_principal_components(pc1_data, pc3_data, 1, 3, colors, size=point_size/16)
        fig.show()
    
def task2(images, images_flattened, labels):    
    # First compute PCs with flattened images
    subclass_images = []
    subclass_images_idx = []
    for i, image in enumerate(images_flattened):
        if labels[i] == DIGIT:
            subclass_images.append(image)
            subclass_images_idx.append(i)
    
    N_IMAGES = len(subclass_images)
    model, results = principal_component_analysis(subclass_images, var_percentage=0.8)
    
    colors_digit = [color_idxs[DIGIT]]*N_IMAGES
    pc1_data = results['PC'].PC1
    pc2_data = results['PC'].PC2
    if PLOT_RES:
        fig, ax = biplot_principal_components(pc1_data, pc2_data, 1, 2, colors=colors_digit, size=point_size, add_legend=False)
        fig.show()
    
    # Now switch to unflattened images to display them later
    sample_idxs = [np.random.randint(0,N_IMAGES) for _ in range(SAMPLE_SIZE)]
    pc_i = 0
    for pc_values in [pc1_data, pc2_data]:
        pc_i += 1
        samples = []
        for idx in sample_idxs:
            samples.append([idx, pc_values[idx]])
        samples.sort(key=lambda x: x[1])
        low_quantile = samples[SAMPLE_SIZE//10]
        high_quantile = samples[(SAMPLE_SIZE//10)*9]

        colors_gray = [color_gray]*N_IMAGES
        colors_gray[low_quantile[0]] = mcolors.CSS4_COLORS['indianred']
        colors_gray[high_quantile[0]] = mcolors.CSS4_COLORS['turquoise']

        point_sizes = [point_size]*N_IMAGES
        point_sizes[low_quantile[0]] = plt.rcParams['lines.markersize']**2
        point_sizes[high_quantile[0]] = plt.rcParams['lines.markersize']**2

        fig, ax = biplot_principal_components(pc1_data, pc2_data, 1, 2, colors_gray, size=point_sizes, add_legend=False)
        ax.set_title(f"Quantile juxtaposition for PC {pc_i}")
        fig.savefig(f"Latex Report/Figures/DIGIT{DIGIT}_SAMPLED_QUANTILES_PC{pc_i}_{int(time.time())}.png")

        plt.clf()
        plt.imsave(f"Latex Report/Figures/DIGIT{DIGIT}_PC{pc_i}_LOW_QUANTILE_{int(time.time())}.png", images[subclass_images_idx[low_quantile[0]]], cmap='gray', vmin=0, vmax=255)
        plt.imsave(f"Latex Report/Figures/DIGIT{DIGIT}_PC{pc_i}_HIGH_QUANTILE_{int(time.time())}.png", images[subclass_images_idx[high_quantile[0]]], cmap='gray', vmin=0, vmax=255)

def task3(images_flattened, colors, n_neighbors = N_NEIGHBORS, min_dist = MIN_DIST, spread = SPREAD, minkowski_p = MINKOWSKI_P):
    umap_model = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        metric = METRIC,
        metric_kwds = {"p": minkowski_p},
        spread = spread
    )
    embedding = umap_model.fit_transform(images_flattened)
    plt.scatter(embedding[:,0],embedding[:,1], c=colors, s=point_size/40)
    plt.title("UMAP embedding of the MNIST Digits dataset")
    plt.legend(handles=legend_patches)
    if PLOT_RES:
        plt.show()
    else:
        plt.savefig(f"Latex Report/Figures/UMAP_PROJECTION_COMPLETE_" \
                f"NEIGHBOURS{n_neighbors}_MINDIST{min_dist}_SPREAD{spread}_MINKOWSKI_P{minkowski_p}.png")
    plt.clf()

    # Obtain PCAs
    model, results = principal_component_analysis(images_flattened, var_percentage=0.8)
    for pc_size in [10, 20, 40]:  
        pca_embedding = None
        if pc_size == 10:
            pca_embedding = np.array([
                results['PC'].PC1,
                results['PC'].PC2,
                results['PC'].PC3,
                results['PC'].PC4,
                results['PC'].PC5,
                results['PC'].PC6,
                results['PC'].PC7,
                results['PC'].PC8,
                results['PC'].PC9,
                results['PC'].PC10
            ]).T
        elif pc_size == 20:
            pca_embedding = np.array([
                results['PC'].PC1,
                results['PC'].PC2,
                results['PC'].PC3,
                results['PC'].PC4,
                results['PC'].PC5,
                results['PC'].PC6,
                results['PC'].PC7,
                results['PC'].PC8,
                results['PC'].PC9,
                results['PC'].PC10,
                results['PC'].PC11,
                results['PC'].PC12,
                results['PC'].PC13,
                results['PC'].PC14,
                results['PC'].PC15,
                results['PC'].PC16,
                results['PC'].PC17,
                results['PC'].PC18,
                results['PC'].PC19,
                results['PC'].PC20
            ]).T
        else:
            pca_embedding = np.array([
                results['PC'].PC1,
                results['PC'].PC2,
                results['PC'].PC3,
                results['PC'].PC4,
                results['PC'].PC5,
                results['PC'].PC6,
                results['PC'].PC7,
                results['PC'].PC8,
                results['PC'].PC9,
                results['PC'].PC10,
                results['PC'].PC11,
                results['PC'].PC12,
                results['PC'].PC13,
                results['PC'].PC14,
                results['PC'].PC15,
                results['PC'].PC16,
                results['PC'].PC17,
                results['PC'].PC18,
                results['PC'].PC19,
                results['PC'].PC20,
                results['PC'].PC21,
                results['PC'].PC22,
                results['PC'].PC23,
                results['PC'].PC24,
                results['PC'].PC25,
                results['PC'].PC26,
                results['PC'].PC27,
                results['PC'].PC28,
                results['PC'].PC29,
                results['PC'].PC30,
                results['PC'].PC31,
                results['PC'].PC32,
                results['PC'].PC33,
                results['PC'].PC34,
                results['PC'].PC35,
                results['PC'].PC36,
                results['PC'].PC37,
                results['PC'].PC38,
                results['PC'].PC39,
                results['PC'].PC40
            ]).T

        umap_model = umap.UMAP(
            n_neighbors = n_neighbors,
            min_dist = min_dist,
            metric = METRIC,
            metric_kwds = {"p": minkowski_p},
            spread = spread
        )
        embedding = umap_model.fit_transform(pca_embedding)
        plt.scatter(embedding[:,0],embedding[:,1], c=colors, s=point_size/4)
        plt.title(f"UMAP embedding of the first {pc_size} PCs of the MNIST Digits")
        plt.legend(handles=legend_patches)
        if PLOT_RES:
            plt.show()
        else:
            plt.savefig(f"Latex Report/Figures/UMAP_PROJECTION_PC_EMBEDDING_" \
                    f"NEIGHBOURS{n_neighbors}_MINDIST{min_dist}_SPREAD{spread}_MINKOWSKI_P{minkowski_p}_PCSIZE{pc_size}.png")
        plt.clf()
    
def main():
    if not PLOT_RES:
        print("No figures will be plotted. Set PLOT_RES to True to enable this function.")

    images = idx2numpy.convert_from_file("data2forEx8+/train-images.idx3-ubyte")
    labels = idx2numpy.convert_from_file("data2forEx8+/train-labels.idx1-ubyte")

    colors = [color_idxs[label] for label in labels]

    N_IMAGES = images.shape[0]
    images_flattened = np.reshape(images, (N_IMAGES,-1))

    ### TASK 1
    if EXECUTE_TASK1:
        task1(images_flattened, colors)
    else:
        print("The code for task 1 will not be executed. Set EXECUTE_TASK1 to True to enable this function.")
    
    ### TASK 2
    if EXECUTE_TASK2:
        task2(images, images_flattened, labels)
    else:
        print("The code for task 2 will not be executed. Set EXECUTE_TASK2 to True to enable this function.")

    
    ### TASK 3
    if EXECUTE_TASK3:
        for nn in [5,15,50]:
            for md in [0.01,0.1,1]:
                for sp in [0.1,1,10]:
                    if md > sp:
                        continue
                    for p in [2,3,5]:
                        task3(images_flattened, colors,
                                  n_neighbors = nn,
                                  min_dist = md,
                                  spread = sp,
                                  minkowski_p = p)
    else:
        print("The code for task 3 will not be executed. Set EXECUTE_TASK3 to True to enable this function.")
    
    return 0
    

if __name__=="__main__":
    main()