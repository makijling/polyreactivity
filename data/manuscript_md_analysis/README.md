# A Repository for Storing Manuscript-Associated Scripts
An open repository containing molecular dynamics (MD) simulation analysis scripts used for a few manuscripts.

# Biochemical and biophysical characterization of natural polyreactivity in antibodies
More precisely Borowska & Boughter et al. Cell Reports 2023. All of the scripts relevant for recreating this analysis can be found in the polyreact_manuscript directory.

get_metrics.py contains the code for creating the processed data from raw MD trajectories (data available upon request).

pyemma_tICA.ipynb is a Jupyter notebook for creating time-lagged independent component analysis data and figures from raw MD trajectories.

generate_figures.ipynb is a Jupyter notebook for recreating figures and statistical analysis shown in the manuscript from processed data.

This processed data can be downloaded from Zenodo (https://doi.org/10.5281/zenodo.8347008) with the data size totalling 1GB.

# Characterization of Shark CD1 Establishes Its Presence in the Primordial MHC
More precisely Almeida & Castro et al. Nature Communications (in review). All scripts relevant for recreating this analysis can be found in the ufa_manuscript directory.

Unfortunately these scripts rely more heavily on the simulated data, which are available upon request. These scripts are provided more for experts to understand how the figures were created.

Both the analyze_AllCd1.py and cd1_ufa_analyze.ipynb files contain code to recreate the analysis, either in script or in Jupyter notebook form.

