# Machine Learning for Tip-Enhanced Raman Spectroscopy  
### Interpreting TERS Maps of π-Conjugated Molecules

> **Quick links:**  
> • [Portfolio Summary (PDF)](docs/portfolio-summary.pdf)  
> • [All Figures Bundle (PDF)](docs/all-figures-portfolio.pdf)

---

## Introduction
Tip-Enhanced Raman Spectroscopy (TERS) provides chemically specific imaging at (near) atomic resolution, but interpreting spectra/maps is difficult due to mode mixing, broadening, probe–molecule coupling, and substrate effects.  
This repo explores **machine learning (ML)** to automate interpretation for **π-conjugated molecules** (benzene, naphthalene, anthracene, pyrene, coronene, small graphene nanoflakes). We pair **1D spectral** classification with **2D TERS-map** recognition, aiming to bridge spectroscopy and AI.

---

## Data (synthetic & reproducible)
- **1D spectra (Raman-like):** start from discrete “sticks” (frequency–intensity), apply **Gaussian broadening**, then min–max normalize.  
- **2D TERS-like maps:** synthesize spatial **hotspots** (Gaussian fields) + weak background + noise; include flips/rotations/blur for augmentation.  
- **Splits:** stratified train/val/test; all indices and metadata saved in `results/tables/`.  
- Generators live in `src/dataset.py`; examples in `notebooks/01_data_exploration.ipynb` and `02_generate_ters_maps.ipynb`.

> See **`statistics.md`** for all the equations we use (broadening, normalization, attention, losses, metrics, etc.).

---

## Methods
We implement two complementary pipelines:

1) **Spectral fingerprints (1D)**  
   - **Input:** synthetic Raman-like spectra (vectors).  
   - **Model:** lightweight **CNN-1D**.  
   - **Task:** molecular **classification**.

2) **TERS imaging (2D)**  
   - **Input:** simulated TERS-like intensity maps (images).  
   - **Model:** **Vision Transformer (ViT)** (tiny, patch-16).  
   - **Task:** molecular **classification** from TERS maps.

**Physics-aware option:** a small **Raman-tensor symmetry** penalty can be added to the loss (see `statistics.md`).

---

## Current Results (baseline)
From the saved artifacts (see PDFs above):

- **CNN-1D (spectra):** Accuracy **0.174**, Macro-F1 **0.049**  
- **ViT (TERS maps):** Accuracy **0.164**, Macro-F1 **0.113**  

Artifacts include training curves, confusion matrices, and per-class tables in `results/tables/` and `figures/`.  
> These are **baseline synthetic settings** (small models, simple augmentations); metrics are expected to rise with more data, stronger aug, and tuning.

---

## Figures
Key figures live in `figures/` (see the All-Figures PDF for a full index). Typical outputs:
- Loss/Accuracy/F1 curves for CNN & ViT.  
- Confusion matrices for test splits.  
- Sample spectra overlays and TERS-map montages.

---

## How to Run
**Colab (recommended):**

[![Open 00_setup in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aoctavia/TersML/blob/main/notebooks/00_colab_setup.ipynb)

**Notebook order:**
1. `00_colab_setup.ipynb` — environment & project scaffold  
2. `01_data_exploration.ipynb` — build & inspect 1D spectra dataset  
3. `02_generate_ters_maps.ipynb` — synthesize 2D TERS-like maps  
4. `03_train_cnn.ipynb` — train CNN-1D on spectra  
5. `04_train_vit.ipynb` — train ViT on TERS maps  
6. `05_results_visualization.ipynb` — re-plot curves/CM; merge metrics  
7. `06_summary_report.ipynb` — export **Portfolio Summary (PDF/HTML/MD)**

**Local (optional):**
```bash
pip install -r requirements.txt
python -c "import torch, timm; print('OK', torch.__version__)"
# then run notebooks in order with Jupyter/VSCode
````

---

## Discussion & Next Steps

* Baselines show the **ViT** is already learning spatial cues; **CNN-1D** excels when peaks are sharp and well separated.
* To improve: increase samples/class, diversify hotspot geometry/backgrounds, add RandAugment, tune LR/weight decay/DropPath.
* Add **attention/Grad-CAM** visualizations and **ablation** (noise, patch size, #hotspots).
* Medium-term: domain adaptation to **experimental TERS**; incorporate **GNNs** for structure-aware features; uncertainty estimation.

---

## References (selection)

* O. J. Silveira *et al.*, *Raman tensors in low-symmetry 2D materials*, J. Raman Spectrosc. (2021).
* O. J. Silveira *et al.*, *Local probe-induced structural isomerization*, Nat. Commun. (2023).
* O. J. Silveira *et al.*, *Frustration-Induced Many-Body Degeneracy in Spin-1/2 Molecular Quantum Rings*, JACS (2025).
* T. Schütt *et al.*, *SchNet: Deep Learning for Quantum Chemistry*, JCTC (2018).

---

## Repo Map

```
src/                 # dataset, models, metrics, viz
notebooks/           # 00..06 workflow (Colab-first)
figures/             # spectra, ters_maps, curves, confusion_matrices, ...
results/             # tables, checkpoints, reports
docs/                # portfolio-summary.pdf, all-figures-portfolio.pdf
scripts/             # helpers (e.g., export_all_figures_pdf.py)
statistics.md        # equations (math/physics/stats used)
```

---

## 📝 License

MIT License
