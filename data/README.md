
# Data

This folder contains preprocessed diagonal segment feature 
matrices extracted from each dataset after z-score 
normalization, pairwise DTW computation (15 stratified 
pairs), and diagonal segment extraction with lmax=3.

## Files

| File | Dataset | Segments | Features |
|------|---------|----------|---------|
| gaitdb_segments_features.csv | GaitDB | 5,602 | L, ell, d, t0, p_l |
| gaitndd_segments_features.csv | GaitNDD | 1,610 | L, ell, d, t0, p_l |

## Raw Datasets

Raw datasets must be downloaded separately:
- GaitDB: https://physionet.org/content/gait-maturation-db/1.0.0/
- GaitNDD: https://physionet.org/content/gaitndd/1.0.0/
- BLISS: https://researchdata.bath.ac.uk/1425/
