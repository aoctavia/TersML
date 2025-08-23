# -*- coding: utf-8 -*-
# See notebook cell for the latest version. Run with: python scripts/export_all_figures_pdf.py
# Requires: reportlab, pillow, pandas

import re
from textwrap import wrap
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIRS = [
    PROJECT_ROOT / "figures" / "spectra",
    PROJECT_ROOT / "figures" / "ters_maps",
    PROJECT_ROOT / "figures" / "curves",
    PROJECT_ROOT / "figures" / "confusion_matrices",
    PROJECT_ROOT / "figures" / "examples",
    PROJECT_ROOT / "figures",
]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DOCS_DIR   = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = DOCS_DIR / "all-figures-portfolio.pdf"
EXTS = (".png", ".jpg", ".jpeg", ".webp")

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def main():
    files = []
    for d in FIG_DIRS:
        if d.exists():
            files += [p for p in sorted(d.rglob("*"), key=natural_key) if p.suffix.lower() in EXTS]
    seen, uniq = set(), []
    for p in files:
        r = p.resolve()
        if r not in seen:
            seen.add(r); uniq.append(p)
    print(f"Found {len(uniq)} image(s).")

    c = canvas.Canvas(str(PDF_PATH), pagesize=A4)
    W, H = A4

    # Cover
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(W/2, H-2*cm, "TERS + ML — All Figures Portfolio")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, H-3*cm, f"Total images: {len(uniq)}")
    sum_path = TABLES_DIR / "summary_models.csv"
    if sum_path.exists():
        try:
            df = pd.read_csv(sum_path)
            y = H-4.2*cm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2*cm, y, "Metrics Summary")
            y -= 0.6*cm
            c.setFont("Helvetica", 10)
            for _, r in df.iterrows():
                line = f"- {r['model']}: ACC={float(r['test_acc']):.3f}, F1-macro={float(r['test_f1_macro']):.3f}"
                for wline in wrap(line, width=95):
                    c.drawString(2*cm, y, wline); y -= 0.5*cm
                    if y < 2*cm:
                        c.showPage(); W, H = A4; y = H-2*cm
        except Exception as e:
            print("Metrics summary read failed:", e)
    c.showPage()

    # Index
    c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, H-2*cm, "Figure Index")
    c.setFont("Helvetica", 10); y = H-3*cm
    for i, p in enumerate(uniq, 1):
        line = f"{i:03d}. {p.relative_to(PROJECT_ROOT)}"
        for wline in wrap(line, width=95):
            c.drawString(2*cm, y, wline); y -= 0.5*cm
            if y < 2*cm:
                c.showPage(); c.setFont("Helvetica", 10); y = H-2*cm

    # Figures
    for i, p in enumerate(uniq, 1):
        try:
            with Image.open(p) as im:
                iw, ih = im.size
            page = landscape(A4) if iw > ih else A4
            c.setPageSize(page); W, H = page
            margin = 1.5*cm; maxw, maxh = W-2*margin, H-3*cm
            scale = min(maxw/iw, maxh/ih); w, h = iw*scale, ih*scale
            x, y = (W-w)/2, (H-h)/2
            c.drawImage(str(p), x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
            c.setFont("Helvetica", 9)
            cap = f"Figure {i:03d}: {p.relative_to(PROJECT_ROOT)}"
            ycap = 1.0*cm
            for wline in wrap(cap, width=110)[:3]:
                c.drawString(margin, ycap, wline); ycap += 0.4*cm
            c.showPage()
        except Exception as e:
            print("Skipping", p, ":", e)

    c.save()
    print("Saved PDF:", PDF_PATH.resolve())

if __name__ == "__main__":
    main()
