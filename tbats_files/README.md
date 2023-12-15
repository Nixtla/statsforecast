
TBATS ADDITIONAL FILES 
======================
This folder contains additional files for the TBATS model. 
- `tbats_main.R` contains the R function used to create one TBATS model. 
- `tbats_functions.R` contains functions needed to run `tbats_main.R`.
- `tbats.py` contains the first complete version of the TBATS model. 
- `models.ipynb` contains the modified version of `models.ipynb` that includes version 1 of the TBATS model. 
- `ISSM.pdf` and `How to select the number of harmonics.pdf` contain part of the research that was done for the TBATS model. The second file is particulary important since it explains how the number of harmonics is selected, a method that was described in the original paper but was not implemented in the R package nor in the Python package (https://github.com/intive-DataScience/tbats/tree/master). 

STATUS
=======
There are two versions of the TBATS model: 
1. The one under tbats_files/ was the first complete version. 
2. The one under nbs/src is the second version. It uses the method described in `How to select the number of harmonics.pdf`. This version is still in development and is incomplete, but has important improvements over the first version.

