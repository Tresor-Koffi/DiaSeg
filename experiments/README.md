# Experiments

This folder contains all scripts to reproduce the 
paper results.

## Scripts

| Script | Reproduces |
|--------|-----------|
| run_clustering_experiments.py | Table 4, Figure 3 |
| ablation_lmax_gaitdb.py | Table 3 |
| bliss_segments_patient_aggregated.py | Tables 5, 6 |

## How to Run

```bash
# From the root directory DiaSeg/

# Table 3 - ablation study
python experiments/ablation_lmax_gaitdb.py

# Table 4 + Figure 3 - clustering
python experiments/run_clustering_experiments.py

# Tables 5, 6 - patient level
python experiments/bliss_segments_patient_aggregated.py
```
