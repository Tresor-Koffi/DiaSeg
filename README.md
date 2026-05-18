# DiaSeg
Diagonal Segment Extraction from DTW Paths for Interpretable Gait Analysis

---

## Datasets

All datasets used in this study are publicly available:

| Dataset | URL |
|---------|-----|
| GaitDB | https://physionet.org/content/gait-maturation-db/1.0.0/ |
| GaitNDD | https://physionet.org/content/gaitndd/1.0.0/ |
| BLISS | https://researchdata.bath.ac.uk/1425/ |

The preprocessed segment feature matrices (z-score normalized, 
15 stratified pairs, lmax=3) are available in the `data/` 
folder, enabling full reproduction of all results without 
requiring access to the raw datasets.

---

## Installation

```bash
git clone https://github.com/Tresor-Koffi/DiaSeg.git
cd DiaSeg
pip install -r requirements.txt
```

---

## How to Reproduce Results

### Table 3 — Break tolerance ablation
```bash
python experiments/ablation_lmax_gaitdb.py
```
Expected: lmax=3, Silhouette=0.318, DB=1.336

### Table 4 — Unsupervised clustering + Figure 3
```bash
python experiments/run_clustering_experiments.py
```
Expected: K-means k=2, Silhouette=0.337 (GaitDB), 0.329 (GaitNDD)

### Tables 5 and 6 — Patient-level classification
```bash
python experiments/bliss_segments_patient_aggregated.py
```
Expected: K-means accuracy=75.0%, Silhouette=0.615

---

## Expected Results Summary

| Script | Table | Key Result |
|--------|-------|-----------|
| `ablation_lmax_gaitdb.py` | Table 3 | lmax=3 optimal |
| `run_clustering_experiments.py` | Table 4 | Sil=0.337 GaitDB |
| `bliss_segments_patient_aggregated.py` | Tables 5,6 | 75.0% accuracy |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{koffi2025diaseg,
  title={DiaSeg: Diagonal Segment Extraction from {DTW} 
         Paths for Interpretable Gait Analysis},
  author={Koffi, Tresor Y. and Hidouri, Amel and 
          Legrand, Corentin and Bertaux, Aur{\'e}lie},
  journal={Data Mining and Knowledge Discovery},
  year={2025},
  publisher={Springer Nature}
}
```

---

## License

This project is licensed under the MIT License. 
See [LICENSE](LICENSE) for details.

---

## Contact

Tresor Y. Koffi — tresor.koffi@u-bourgogne.fr  
Université Bourgogne Europe, CIAD UR 7533  
21000 Dijon, France
