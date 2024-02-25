*** Programming environment ***
The entire code was independently written in Python 3.10.13. The packages were installed using a plain virtual environment generated by miniconda3. To install the required packages that are not provided along the Python Standard Library, execute the command ```pip install -r requirements.txt``` in this folder. 

*** Packages and their purpose ***
The following packages are used in the code:
- matplotlib.pyplot: Plot functions
- matplotlib.colors: CCS colors for plots
- matplotlib.patches: Manually defined legends for plots
- numpy: Data processing
- idx2numpy: Convert idx data to a numpy array
- pca: Perform Principal Component Analysis
- umap: Perform Uniform Manifold Approximation
- time: Generate unique figure filenames

*** Hyperparameters and their abbreviations ***
The hyperparameters and their abbreviations are specified in the following. The abbreviations are used in the generated figures to identify the experiment parameters for each result.
- N_NEIGHBOURS: Number of neighbours on which to base the fuzzy topology
- MIN_DIST: Minimum distance between elements to be connected
- SPREAD: Spread coefficient
- METRIC: Distance metric
- MINKOWSKI_P: P value for Minkowski distance metric

*** Figures ***
I did not add the raw figures folder and the source code of the experiment report due to limitations in data transfer via mail.

*** File and directory overview ***
- requirements.txt: Lists the package names imported into the Python script
- dimreduc.py: Python script containing the PCA and UMAP experiments 
- Experiment Report HW8.pdf: Experiment report