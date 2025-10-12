# Mathematics & Physics Reference for the TERS-ML Portfolio

This document summarizes the key equations used in this portfolio: synthetic Raman spectra (1D), TERS-like map generation (2D), learning objectives, optimization, and evaluation metrics.

---

## 1) Spectral Grid, Sticks, and Gaussian Broadening

**Spectral grid (Raman shift, in cm\$^{-1}\$).**
Let \$\nu\_{\min}, \nu\_{\max}, \Delta\nu\$ be min, max, and step. For \$L\$ points:

$$
g_k = \nu_{\min} + k\,\Delta\nu,\quad k=0,\dots,L-1,\qquad
\Delta\nu = \frac{\nu_{\max}-\nu_{\min}}{L-1}.
$$

**Stick spectrum \$\to\$ Gaussian-broadened spectrum.**
Given sticks \${(\nu\_i, I\_i)}\_{i=1}^M\$,

$$
S(\nu) = \sum_{i=1}^{M} I_i \,\mathcal{G}_\sigma(\nu-\nu_i),
\qquad
\mathcal{G}_\sigma(\nu) = \frac{1}{\sigma\sqrt{2\pi}}
\exp\!\Big(-\frac{\nu^2}{2\sigma^2}\Big).
$$

On the discrete grid, \$S\_k = S(g\_k)\$.

**Convolution view.** If \$X(\nu)=\sum\_i I\_i,\delta(\nu-\nu\_i)\$, then \$S = X \* \mathcal{G}\_\sigma\$.

**FWHM–\$\sigma\$ (Gaussian).**

$$
\mathrm{FWHM} = 2\sqrt{2\ln 2}\,\sigma \approx 2.35482\,\sigma.
$$

**Min–max normalization (used here).**

$$
\tilde{S}_k=\frac{S_k-\min_j S_j}{\max_j S_j-\min_j S_j+\varepsilon},
\qquad \varepsilon>0~\text{(tiny)}.
$$

---

## 2) Synthetic TERS-Like Map Generation (2D)

We synthesize intensity images on a grid \$(x,y)\in{0,\dots,H-1}\times{0,\dots,W-1}\$ by summing localized hotspots plus weak background and noise:

$$
I(x,y) = \mathrm{clip}\!\left(
\sum_{j=1}^{J} A_j \exp\!\Big(
-\frac{(x-x_j)^2+(y-y_j)^2}{2\sigma_j^2}
\Big) + b + \eta(x,y),~
0,~1\right),
$$

where \$A\_j\in\[0,1]\$ (amplitude), \$\sigma\_j>0\$ (width), \$(x\_j,y\_j)\$ hotspot centers, \$b\ge 0\$ background, and \$\eta\sim\mathcal{N}(0,\sigma\_n^2)\$ pixel noise.

**Augmentations.**

* Horizontal/vertical flips: \$I'(x,y)=I(H!-!1!-!x,y)\$ or \$I'(x,y)=I(x,W!-!1!-!y)\$.
* \$90^\circ\$ rotations: \$I'=\mathrm{rot}\_k(I)\$, \$k\in{0,1,2,3}\$.
* Gaussian blur (separable):

$$
k[n]=\frac{1}{Z}\exp\!\Big(-\frac{n^2}{2\sigma_b^2}\Big),\qquad
I' = (I * k) * k^\top.
$$

**TERS enhancement (heuristic context).**

$$
\mathrm{EF}_{\mathrm{TERS}}(\mathbf{r})~\propto~
|E(\mathbf{r},\omega_i)|^2~|E(\mathbf{r},\omega_s)|^2,
$$

with \$E\$ the local near-field at incident/scattered \$\omega\_i,\omega\_s\$.
(Here we emulate spatial variation via Gaussian hotspots.)

---

## 3) Raman Scattering (Tensor Form, Context)

For mode \$m\$ with Raman tensor \$\mathbf{R}\_m\$ and incident/scattered polarization \$\mathbf{e}\_i,\mathbf{e}\_s\$,

$$
I_m \propto \big|\mathbf{e}_s^\top\,\mathbf{R}_m\,\mathbf{e}_i\big|^2\,\delta(\nu-\nu_m).
$$

A simple symmetry regularizer (if a predicted \$\hat{\mathbf{R}}\$ is used):

$$
\mathcal{L}_{\mathrm{sym}} = \|\hat{\mathbf{R}}-\hat{\mathbf{R}}^\top\|_F^2.
$$

---

## 4) 1D CNN for Spectra (Convolution & Blocks)

For input \$x\in\mathbb{R}^{C\_{\mathrm{in}}\times L}\$, 1D convolution to output channel \$c\$:

$$
y_c[n] = b_c + \sum_{c'=0}^{C_{\mathrm{in}}-1}
\sum_{m=0}^{K-1} W_{c,c'}[m]\;x_{c'}[n+m'],
$$

where \$m'\$ respects padding/stride/dilation.
Typical block: Conv1D \$\rightarrow\$ Norm \$\rightarrow\$ ReLU \$\rightarrow\$ Pool.

**Softmax classifier.**

$$
p_c = \mathrm{softmax}(z)_c=\frac{e^{z_c}}{\sum_{k=1}^K e^{z_k}}.
$$

---

## 5) Vision Transformer (ViT) for 2D TERS Maps

**Patch embedding.**
Image \$X\in\mathbb{R}^{H\times W\times 3}\$ split into \$N\$ patches of size \$P\times P\$.
Flatten each patch \$x\_i\in\mathbb{R}^{3P^2}\$ and project:

$$
\mathbf{z}_i^{(0)} = x_i \mathbf{E} + \mathbf{e}^{\mathrm{pos}}_i,\qquad
\mathbf{E}\in\mathbb{R}^{3P^2\times d}.
$$

Add a class token \$\mathbf{z}^{(0)}\_{\mathrm{cls}}\$.

**Self-attention (single head; MSA is a sum of heads).**

$$
\mathbf{Q}=\mathbf{Z}\mathbf{W}_Q,\quad
\mathbf{K}=\mathbf{Z}\mathbf{W}_K,\quad
\mathbf{V}=\mathbf{Z}\mathbf{W}_V,\qquad
\mathrm{Attn}(\mathbf{Z})=\mathrm{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\Big)\mathbf{V}.
$$

Transformer block stacks: LN \$\to\$ MSA \$\to\$ Residual \$\to\$ LN \$\to\$ MLP \$\to\$ Residual.

**Classification head (use CLS token).**

$$
\hat{\mathbf{y}}=\mathrm{softmax}\big(\mathbf{W}_{\mathrm{cls}}\mathbf{h}_{\mathrm{cls}}+\mathbf{b}\big).
$$

---

## 6) Losses (Classification & Physics-Aware)

**Weighted cross-entropy (class imbalance).**

$$
\mathcal{L}_{\mathrm{CE}}(\mathbf{y},\mathbf{p}) = -\sum_{c=1}^{K}
w_c\, y_c \log p_c,\qquad
\sum_{c} y_c=1.
$$

A common choice: \$w\_c \propto 1/\max(n\_c,1)\$ and normalized to mean \$1\$.

**Label smoothing (optional).**

$$
\tilde{\mathbf{y}}=(1-\varepsilon)\,\mathbf{y}+\frac{\varepsilon}{K}\,\mathbf{1}.
$$

**Physics-informed addition (optional).**

$$
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+\lambda\,\mathcal{L}_{\mathrm{sym}},
\qquad
\mathcal{L}_{\mathrm{sym}}=\|\hat{\mathbf{R}}-\hat{\mathbf{R}}^\top\|_F^2.
$$

---

## 7) Optimization, Schedules, and AMP

**AdamW (decoupled weight decay).**

$$
\begin{aligned}
g_t &= \nabla_\theta \mathcal{L}_t,\\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t},\quad
\hat{v}_t &= \frac{v_t}{1-\beta_2^t},\\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\,\lambda_{\mathrm{wd}}\,\theta_t.
\end{aligned}
$$

**Cosine annealing LR (no restarts).**

$$
\eta(t) = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})
\big(1+\cos(\pi t/T_{\max})\big).
$$

**Automatic Mixed Precision (AMP).**
Forward/backward in FP16/FP32 with dynamic loss scaling.

---

## 8) Data Splits & Sampling

**Stratified split** preserves class proportions.
If the dataset has \$n\_c\$ samples for class \$c\$ and split ratio \$r\$, expected per-split count is \$\lfloor r,n\_c \rceil\$.

**Class weights (used above).** A simple normalization:

$$
w_c = \frac{\sum_{k=1}^{K} n_k}{K\,\max(n_c,1)}\quad \text{so that}\quad
\frac{1}{K}\sum_{c=1}^K w_c = 1.
$$

---

## 9) Metrics

**Per-class precision/recall/F1 (for class \$c\$).**

$$
\mathrm{Precision}_c = \frac{TP_c}{TP_c+FP_c},\qquad
\mathrm{Recall}_c = \frac{TP_c}{TP_c+FN_c},
$$

$$
\mathrm{F1}_c = \frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}
{\mathrm{Precision}_c+\mathrm{Recall}_c+\varepsilon}.
$$

**Macro-F1.** Unweighted mean over classes:

$$
\mathrm{F1}_{\mathrm{macro}} = \frac{1}{K}\sum_{c=1}^{K} \mathrm{F1}_c.
$$

**Accuracy.**

$$
\mathrm{ACC} = \frac{\sum_c TP_c}{\sum_c (TP_c+FP_c)} = \frac{\#\text{correct}}{\#\text{total}}.
$$

**Confusion matrix \$\mathbf{C}\in\mathbb{N}^{K\times K}\$.**

$$
C_{ij} = \#\{\text{samples with true class } i \text{ predicted as } j\}.
$$

---

## 10) Image & Spectrum Transforms

**To-Tensor & Normalize (ViT input).**

$$
\mathbf{X}_{\mathrm{norm}} = \frac{\mathbf{X}-\boldsymbol{\mu}}{\boldsymbol{\sigma}},
\qquad \boldsymbol{\mu}=(0.5,0.5,0.5),\ \boldsymbol{\sigma}=(0.5,0.5,0.5).
$$

(After mapping $\[0,1]\to\[-1,1]\$.)

**1D spectral shift (augmentation):** circular roll by \$s\$ bins:

$$
\tilde{S}[k] = S[(k-s)\bmod L].
$$

**Additive Gaussian noise (1D/2D).**

$$
\tilde{x} = \mathrm{clip}(x + \epsilon),\qquad \epsilon\sim \mathcal{N}(0,\sigma^2).
$$

---

## 11) Notation Recap

* \$K\$: number of classes
* \$L\$: spectrum length (grid points)
* \$H,W\$: image height/width (pixels)
* \$\sigma\$: Gaussian width (broadening or blur)
* \$w\_c\$: class weight for class \$c\$
* \$\lambda\_{\mathrm{wd}}\$: weight decay; \$\lambda\$: physics regularization weight
* \$\eta\$: learning rate; \$T\_{\max}\$: cosine period (epochs/steps)
* \$TP,FP,FN\$: true/false positive/negative counts


  ## Note : some of equations can't show on Readme and not completed, so i will rewrite with other tools

---
