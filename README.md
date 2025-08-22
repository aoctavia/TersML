## 🎯 Portofolio Ideal untuk Project 2 (TERS + ML)

### 🔑 Tujuan Portofolio

* Menunjukkan **coding ability** (Python, PyTorch, GNN/CNN).
* Menunjukkan **pemahaman physics & spectroscopy** (DFT → vibrational/Raman).
* Menunjukkan **applied ML pipeline** (dataset → model → results → visualization).
* Harus *clean, reproducible, and visually clear* → supervisor bisa langsung lihat.

---

## 📂 Struktur Portfolio (GitHub)
```
TERS-ML-Portfolio/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt                 # untuk local/dev; Colab install via cells
├─ notebooks/
│  ├─ 00_colab_setup.ipynb          # install deps (PyTorch Geometric/DeepChem), mount Drive (opsional)
│  ├─ 01_data_exploration_qm9.ipynb # EDA: vibrational modes → synthetic Raman spectra
│  ├─ 02_feature_repr.ipynb         # graph (GNN) vs 1D spectrum (CNN) representations
│  ├─ 03_train_gnn.ipynb            # GNN (SchNet/GraphConv) → predict freq/intensity
│  ├─ 04_train_cnn.ipynb            # 1D CNN → classify spectra / regress intensities
│  ├─ 05_results_visualization.ipynb# overlay spectra, confusion matrix, metrics
│  └─ 06_export_report.ipynb        # export figures & summary (PDF/Markdown)
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py                    # loader QM9 + generator synthetic Raman
│  ├─ features.py                   # fingerprints, graph builders, normalization
│  ├─ models.py                     # GNN (e.g., SchNet/MPNN) & 1D-CNN
│  ├─ train.py                      # training loops + checkpoints
│  ├─ metrics.py                    # MAE/RMSE/Acc/F1, spectral distance (e.g., MSE/DTW)
│  └─ viz.py                        # plotting spectra, confusion matrix, scatter parity
├─ data/
│  ├─ README_DATA.md                # petunjuk ambil QM9 / cache di Colab
│  └─ (auto-generated caches)       # .gitignore akan mengabaikan ini
├─ figures/
│  ├─ examples/                     # overlay spectra (GT vs ML)
│  ├─ confusion_matrices/
│  └─ curves/                       # loss/metric curves
├─ results/
│  ├─ logs/                         # tensorboard / csv logs
│  ├─ checkpoints/                  # model weights (opsional, besar → rilis)
│  └─ tables/                       # metrics & ablations csv
├─ refs/
│  ├─ refs.bib                      # sitasi Raman/TERS/DFT/ML
│  └─ reading_list.md               # 5–10 paper kunci + 1–2 kalimat takeaway
├─ scripts/
│  ├─ prepare_qm9.py                # unduh/preprocess (opsional untuk lokal)
│  ├─ gen_synthetic_raman.py        # buat spek Raman dari vibrational modes
│  └─ export_report.py              # gabung hasil → md/pdf (opsional)
├─ docs/
│  ├─ portfolio-summary.pdf         # mini-paper 2–3 halaman (opsional)
│  └─ template.tex                  # template LaTeX ringkas (Elsevier-like)
└─ CITATION.cff                     # metadata sitasi repo (opsional)
```

---

## 📖 Isi README.md

### Title

**Machine Learning for Raman Spectroscopy: Predicting and Interpreting Molecular Spectra**

### Abstract

> In this project, we explore how machine learning can accelerate the interpretation of Raman/TERS spectra at the nanoscale. Using vibrational data from QM9 molecules, we compare graph neural networks (GNNs) and convolutional neural networks (CNNs) for predicting vibrational modes and classifying molecules from synthetic Raman spectra. This work demonstrates how ML can enhance molecular structure discovery, in line with recent developments in computational spectroscopy.

### Sections

1. **Introduction**

   * Why Raman/TERS is important for nanoscale imaging.
   * Role of ML in interpreting complex spectroscopy.

2. **Dataset**

   * QM9 / vibrational frequencies from DFT.
   * Preprocessing → spectra representation.

3. **Methods**

   * Graph Neural Networks (molecule as graph).
   * CNNs (spectra as 1D signal).
   * Training setup (loss, optimizer, metrics).

4. **Results**

   * Regression: predicted vs reference spectra.
   * Classification: molecule identity from spectra.
   * Visualization: spectra overlays, confusion matrix.

5. **Discussion**

   * Strengths/weaknesses of ML vs DFT.
   * Future directions (transfer to real TERS).

6. **References**

---

## 📊 Expected Deliverables (Menarik Supervisor)

* **Plots**:

  * Sample molecule (benzene, etc.) with reference vs ML-predicted spectrum.
  * Confusion matrix for classification.
  * Training curves (loss vs epochs).
* **Code snippets**: reproducible, simple to run in Colab.
* **Short PDF summary** (like a mini-paper): 2–3 pages.

---

## 🚀 Roadmap (Step-by-Step)

1. **EDA:** Ambil subset QM9 (100–500 molecules) → visualize vibrational spectra.
2. **Baseline ML:** Train simple CNN to classify molecules from their spectra.
3. **Graph ML:** Use GNN (e.g., PyTorch Geometric) → predict frequencies/intensities.
4. **Comparison:** Show CNN vs GNN performance.
5. **Wrap-up:** Make a clean GitHub repo + Colab demo.

---

📌 Dengan portofolio ini, kamu menunjukkan:

* Kamu **bisa coding** (ML pipeline + visualization).
* Kamu **mengerti spectroscopy** (walau tidak eksperimen langsung).
* Kamu siap untuk **mendalami TERS** saat PhD.
