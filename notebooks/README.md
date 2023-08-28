This folder contains jupyter notebooks with the analysis code. Counting and visualization helper functions can
be found in [tools](https://github.com/drstrupf/clinical-trial-landscape/tree/main/tools).

### Preprocessing
The notebook ``01_preprocessing.ipynb`` contains all steps from a raw ``.csv``
[ClinicalTrials.gov](https://clinicaltrials.gov/) download file to a
long-form dataset of clinical trials for multiple sclerosis.

### Analysis
The notebook ``02_data_analysis.ipynb`` contains the filtering of the intermediate data to only contain
industry funded interventional drug trials with detailed location listing, the preprocessing and integration
of socioeconomic data from [Natural Earth](https://www.naturalearthdata.com/), and the preprocessing 
and integration of Human Development Index data from [UNdata](http://data.un.org/Default.aspx). This 
notebook also contains all steps to obtain trial and trial site counts for different regional and 
socioeconomic strata.

### Heat maps
The notebook ``03_heatmaps.ipynb`` contains various heat map visualizations of the trial and trial site
counts generated in the previous step.

### World maps
The notebook ``04_worldmaps.ipynb`` shows how to plot trial data on a world map.

### Regression
The notebook ``05_regression.ipynb`` contains a linear regression of the disproportionality in trial
sites as function of the Human Development Index.

### Figures
The notebook ``06_figures_for_publication.ipynb`` contains the code that was used to generate the
main figures for this project.