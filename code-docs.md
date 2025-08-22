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
