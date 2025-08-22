```
TERS-ML-Portfolio/
│── README.md                  # judul + abstrak + instruksi run (sudah saya draftkan)
│── LICENSE
│── .gitignore
│── requirements.txt            # untuk local/dev
│
├── notebooks/                  # inti: step-by-step colab notebooks
│   ├── 00_colab_setup.ipynb          # setup environment + struktur folder
│   ├── 01_data_exploration.ipynb     # EDA vibrational dataset (π-conjugated molecules)
│   ├── 02_generate_ters_maps.ipynb   # buat TERS-like spectra & 2D maps
│   ├── 03_train_cnn.ipynb            # train CNN untuk 1D spectra
│   ├── 04_train_vit.ipynb            # train Vision Transformer untuk 2D TERS maps
│   ├── 05_results_visualization.ipynb# visualisasi hasil (overlay, confusion matrix, attention)
│   └── 06_summary_report.ipynb       # ringkasan hasil + export ke pdf/markdown
│
├── src/                       # modular python code
│   ├── __init__.py
│   ├── dataset.py              # loader QM9 subset + π-conjugated molecules + generator TERS maps
│   ├── features.py             # graph rep, fingerprints, spectral preprocessing
│   ├── models.py               # CNN (1D), ViT (2D), physics-informed loss functions
│   ├── train.py                # training loop, checkpoint, early stopping
│   ├── metrics.py              # MAE, RMSE, F1, spectral distance, tensor symmetry loss
│   └── viz.py                  # plotting spectra, overlays, confusion matrix, attention maps
│
├── data/                      # dataset kecil (dummy / preprocessed)
│   ├── README_DATA.md          # petunjuk unduh QM9 / generate vibrational modes
│   └── molecules.json          # contoh molekul (benzene, naphthalene, anthracene…)
│
├── figures/                   # output figure
│   ├── spectra/                # overlay DFT vs ML spectra
│   ├── ters_maps/              # contoh TERS-like images
│   ├── confusion_matrices/
│   └── attention/              # attention heatmaps ViT
│
├── results/
│   ├── logs/                   # training logs
│   ├── checkpoints/            # saved models
│   └── tables/                 # metrics csv, ablation studies
│
├── refs/                       # referensi ilmiah
│   ├── refs.bib                 # bibtex untuk citasi
│   └── reading_list.md          # 5–10 paper Silveira + ML spectroscopy
│
└── docs/                       # optional dokumentasi
    ├── portfolio-summary.pdf    # mini-paper (2–3 halaman)
    └── poster.png               # visual summary untuk CV/interview
```


---

# Step 0 — `00_colab_setup.ipynb`

| Cell | Title                             | Main Actions                                                     | Outputs/Artifacts                                                          |
| ---- | --------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1    | Title & Goals                     | State objectives of the setup                                    | —                                                                          |
| 2    | Check Runtime & GPU               | Print Python/OS/GPU/CUDA info                                    | version logs                                                               |
| 3    | Install Core Packages             | Torch, Lightning, RDKit, timm                                    | ready environment                                                          |
| 4    | (Optional) PyG                    | Attempt `torch-geometric` install                                | (optional) PyG                                                             |
| 5    | Verify Torch/CUDA                 | Check `torch.cuda.is_available()`                                | device log                                                                 |
| 6    | Create Folder Structure           | `data/`, `src/`, `figures/`, `results/`, etc.                    | project tree                                                               |
| 7    | `.gitignore` & `requirements.txt` | Write files                                                      | `.gitignore`, `requirements.txt`                                           |
| 8    | Stub `src/`                       | `dataset.py`, `features.py`, `models.py`, `metrics.py`, `viz.py` | files under `src/`                                                         |
| 9    | `refs/` & data README             | `refs.bib`, `data/README_DATA.md`                                | refs files                                                                 |
| 10   | Set Seeds & Plot Style            | Global seeding + matplotlib style                                | reproducibility                                                            |
| 10.5 | **Fix PYTHONPATH**                | Add project root to `sys.path`, `chdir`                          | safe `src.*` imports                                                       |
| 11   | Sanity Check                      | Generate 1D spectrum & 2D TERS map                               | `figures/spectra/spectrum_demo.png`, `figures/ters_maps/ters_map_demo.png` |
| 12   | Mount Drive (opt.)                | Option to save into Google Drive                                 | —                                                                          |
| 13   | Wrap-up                           | Next steps note                                                  | —                                                                          |

---

# Step 1 — `01_data_exploration.ipynb`

| Cell | Title                | Main Actions                   | Outputs/Artifacts                                                                            |
| ---- | -------------------- | ------------------------------ | -------------------------------------------------------------------------------------------- |
| 1    | Title & Goals        | Define EDA scope for spectra   | —                                                                                            |
| 2    | Imports & Paths      | Set `PROJECT_ROOT` & dirs      | path check                                                                                   |
| 3    | Seed                 | Reproducibility seed           | —                                                                                            |
| 4    | Helper Imports       | Pull functions from `src/`     | —                                                                                            |
| 5    | Molecule List        | Save `molecules.json`          | `data/molecules.json`                                                                        |
| 6    | Grid & Generator     | Define `synthesize_spectrum_*` | utility funcs                                                                                |
| 7    | Build Dataset        | N samples/class                | `spectra_matrix`, `meta_df`                                                                  |
| 8    | Save Dataset         | NPZ + metadata CSV             | `data/toy_ters_raman_spectra.npz`, `results/tables/eda_meta.csv`                             |
| 9    | Overlay per Molecule | One sample per class plot      | `figures/spectra/overlay_per_molecule.png`                                                   |
| 10   | Batch Grid Plot      | Multiple samples grid          | `figures/spectra/samples_grid.png`                                                           |
| 11   | Basic Stats          | Grouped by molecule            | dataframe                                                                                    |
| 12   | Save & Bar Chart     | Table + sample counts bar      | `results/tables/eda_summary_by_molecule.csv`, `figures/spectra/bar_samples_per_molecule.png` |
| 13   | Histogram Mean Freq  | Distribution of mean vib. freq | `figures/spectra/hist_mean_frequency.png`                                                    |
| 14   | Wrap-up              | Next steps note                | —                                                                                            |
| 15   | (MD) Split Goals     | Explain split plan             | —                                                                                            |
| 16   | Stratified Split 1   | Train vs temp                  | index arrays                                                                                 |
| 17   | Stratified Split 2   | Val vs test + save indices     | `results/tables/split_indices.csv/json`                                                      |
| 18   | Save NPZ Splits      | Split packages                 | `data/spectra_{train,val,test}.npz`                                                          |
| 19   | Split Distribution   | Bar chart per split            | `figures/spectra/split_distribution.png`                                                     |
| 20   | Split Wrap-up        | Next steps                     | —                                                                                            |

---

# Step 2 — `02_generate_ters_maps.ipynb`

| Cell | Title                 | Main Actions                          | Outputs/Artifacts                                               |
| ---- | --------------------- | ------------------------------------- | --------------------------------------------------------------- |
| 1    | Title & Goals         | Define TERS 2D scope                  | —                                                               |
| 2    | Imports & Paths       | Set dirs                              | —                                                               |
| 3    | Seed                  | Reproducibility                       | —                                                               |
| 4    | Helpers & Molecules   | Load `generate_ters_map` & list       | —                                                               |
| 5    | Config & Augmentation | Class parameters; noise/flip/rot/blur | generator funcs                                                 |
| 6    | Generate Dataset      | Produce all images                    | `images (N,H,W)`, `meta_df`                                     |
| 7    | Save Full             | NPZ + meta CSV                        | `data/ters_maps_full.npz`, `results/tables/ters_maps_meta.csv`  |
| 8    | One per Molecule      | Grid of examples                      | `figures/ters_maps/samples_per_molecule.png`                    |
| 9    | Intensity Stats       | mean/std/p99                          | `results/tables/ters_maps_stats.csv`                            |
| 10   | Stratified Split      | Train/val/test + index                | `results/tables/ters_maps_split_indices.csv/json`               |
| 11   | Save NPZ Splits       | Per-split packages                    | `data/ters_maps_{train,val,test}.npz`                           |
| 12   | Dist. & Montage       | Class bars + train montage            | `figures/ters_maps/split_distribution.png`, `train_montage.png` |
| 13   | Wrap-up               | Next steps note                       | —                                                               |

---

# Step 3 — `03_train_cnn.ipynb` (1D spectra)

| Cell | Title                 | Main Actions                                     | Outputs/Artifacts                                                                        |
| ---- | --------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| 1    | Title & Goals         | CNN training scope                               | —                                                                                        |
| 2    | Imports & Paths       | Set figure/checkpoint dirs                       | —                                                                                        |
| 3    | Seed & Device         | AMP flag                                         | —                                                                                        |
| 4    | Load Splits           | `spectra_{train,val,test}.npz` + label enc.      | classes                                                                                  |
| 4b   | **Force float32**     | Avoid dtype mismatch                             | —                                                                                        |
| 5    | Dataset & Aug         | Noise/shift (float32-safe)                       | `SpectraDataset`                                                                         |
| 6    | DataLoaders & Weights | Class weights                                    | —                                                                                        |
| 7    | Model & Optim         | `SimpleCNN1D`, CE+weights, AdamW, Cosine         | —                                                                                        |
| 8    | Train/Eval Loop       | AMP (new API), scaler                            | `run_epoch`                                                                              |
| 9    | Training Main         | ES on val F1; save best                          | `results/checkpoints/cnn1d_best.pt`, `results/tables/cnn1d_training_history.csv`         |
| 10   | Plot Curves           | Loss/Acc/F1                                      | `figures/curves/cnn1d_*.png`                                                             |
| 11   | **Load & Test**       | `torch.load(..., weights_only=False)`, test eval | `results/tables/cnn1d_test_report.csv`, `results/tables/cnn1d_confusion_matrix.npy/json` |
| 12   | Plot CM               | Confusion matrix                                 | `figures/confusion_matrices/cnn1d_confusion_matrix.png`                                  |
| 13   | Config Artifacts      | Save classes & config                            | `results/tables/cnn1d_*.json`                                                            |
| 14   | Demo Inference        | Sample predictions plot                          | `figures/curves/cnn1d_inference_examples.png`                                            |
| 15   | Wrap-up               | Artifacts & next                                 | —                                                                                        |

---

# Step 4 — `04_train_vit.ipynb` (2D TERS maps)

| Cell | Title                 | Main Actions                                  | Outputs/Artifacts                                             |
| ---- | --------------------- | --------------------------------------------- | ------------------------------------------------------------- |
| 1    | Title & Goals         | ViT training scope                            | —                                                             |
| 2    | Imports & Paths       | Set dirs                                      | —                                                             |
| 3    | Seed & Device         | AMP + channels\_last                          | —                                                             |
| 4    | Load Splits           | `ters_maps_{train,val,test}.npz` + label enc. | classes                                                       |
| 5    | Transforms & Dataset  | Resize→224, aug, ToTensor+Norm                | dataset                                                       |
| 6    | DataLoaders & Weights | Batch/workers/weights                         | —                                                             |
| 7    | Model & Optim         | `timm` ViT tiny, AdamW, Cosine LR             | —                                                             |
| 8    | AMP-safe Loop         | New `autocast` + scaler                       | `epoch_step`                                                  |
| 9    | Training              | ES on val F1; save best                       | `results/checkpoints/vit_best.pt`, `vit_training_history.csv` |
| 10   | Plot Curves           | Loss/Acc/F1                                   | `figures/curves/vit_*.png`                                    |
| 11   | **Load & Test**       | `weights_only=False` + test eval              | `vit_test_report.csv`, `vit_confusion_matrix.npy/json`        |
| 12   | Plot CM               | Confusion matrix                              | `figures/confusion_matrices/vit_confusion_matrix.png`         |
| 13   | Inference Montage     | Test examples grid                            | `figures/ters_maps/vit_inference_montage.png`                 |
| 14   | Artifacts             | Save `vit_weights_only.pt`, classes/config    | files under `results/`                                        |

---

# Step 5 — `05_results_visualization.ipynb`

| Cell | Title                  | Main Actions              | Outputs/Artifacts                                 |
| ---- | ---------------------- | ------------------------- | ------------------------------------------------- |
| 1    | Title & Goals          | Results dashboard scope   | —                                                 |
| 2    | Imports & Paths        | Set dirs                  | —                                                 |
| 3    | Load Metrics & Reports | Load CSV/JSON/NPY         | data objects                                      |
| 4    | Re-plot Curves         | CNN & ViT                 | `figures/curves/*_replot.png`                     |
| 5    | Plot CM                | CNN & ViT                 | `figures/confusion_matrices/*_replot.png`         |
| 6    | Per-class Metrics      | Extract per-class table   | `results/tables/per_class_metrics_cnn_vs_vit.csv` |
| 7    | Grouped F1 Bar         | CNN vs ViT per class      | `figures/curves/per_class_f1_cnn_vs_vit.png`      |
| 8    | Summary Table          | ACC & macro-F1            | `results/tables/summary_models.csv`               |
| 9    | Show Artifacts         | Display inference/CM figs | —                                                 |
| 10   | Export Markdown        | Compact summary           | `results/tables/summary.md`                       |
| 11   | Done & Next            | Checklist                 | —                                                 |

---

# Step 6 — `06_summary_report.ipynb`

| Cell | Title                 | Main Actions                         | Outputs/Artifacts                  |
| ---- | --------------------- | ------------------------------------ | ---------------------------------- |
| 1    | Title & Goals         | Report scope                         | —                                  |
| 2    | Imports & Paths       | Set dirs                             | —                                  |
| 3    | Load Tables & Metrics | Pull artifacts from `results/tables` | data objects                       |
| 4    | Collect Figures       | Candidate figure list                | path list                          |
| 5    | Build Markdown        | Short report                         | `results/tables/summary_report.md` |
| 6    | Install reportlab     | PDF dependency                       | —                                  |
| 7    | PDF Helpers           | Styles, table, fit image             | utils                              |
| 8    | Build PDF             | Portfolio summary                    | `docs/portfolio-summary.pdf`       |
| 9    | Build HTML            | Lightweight page                     | `docs/summary.html`                |
| 10   | Output Recap          | Tips                                 | —                                  |

> Extra utility (outside Steps 0–6): **All-figures exporter** → `docs/all-figures-portfolio.pdf` (cover + index + every figure).

