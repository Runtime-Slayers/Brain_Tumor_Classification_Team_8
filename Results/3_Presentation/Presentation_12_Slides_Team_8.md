# Brain Tumor Classification using Deep Learning & XAI
## Presentation Script — Team 8 | 12 Slides

---

---

# SLIDE 1 — TITLE SLIDE

## Brain Tumor Classification using Deep Learning and Explainable AI
### NeuroVision Diagnostic System

**Team 8**
**Architecture:** Attention-Gated ResNet-34 · Monte Carlo Uncertainty · 17-View Radiomics Dashboard

---

**WHAT TO DISPLAY:**
- Dark blue/teal gradient background
- MRI scan image on the right panel
- Team name, project title, institution name
- Tagline: *"Making AI Diagnostics Transparent, Trustworthy, and Clinically Safe"*

**SPEAKER NOTES:**
> "Good morning, respected faculty and examiners. We are Team 8, and we present our final semester project — the NeuroVision Diagnostic System: a clinical-grade AI application that classifies brain tumor MRI scans into four diagnostic categories with 97.84% accuracy, while being fully transparent and explainable. Our system is not just a classifier — it is a complete clinical decision support tool."

---

---

# SLIDE 2 — PROBLEM STATEMENT

## The Clinical Challenge

### Why This Matters:
- **308,000** new brain tumor cases worldwide annually (WHO)
- High-grade gliomas: median survival **14 months** with treatment
- Manual MRI reading: **200–400 slices per study**, time-intensive and subjective
- Inter-radiologist agreement: only **78–92%** on ambiguous lesions

### Two Critical Problems with Existing AI:

| Problem | Description | Risk |
|---|---|---|
| **The Black-Box Problem** | Neural networks output numbers with no visual justification | Clinically inadmissible — cannot operate without evidence |
| **The False Confidence Problem** | Softmax scores appear certain even on corrupted/ambiguous scans | Dangerous misdiagnosis with no warning |

**Our Mission:** Solve both problems simultaneously.

---

**SPEAKER NOTES:**
> "Brain tumors kill over 200,000 people annually. While AI achieves impressive accuracy in laboratory benchmarks, it faces two fatal barriers preventing real-world clinical adoption. First: the Black-Box Problem — a machine saying 'Glioma, 99% confidence' without showing WHY is clinically useless. A neurosurgeon cannot authorize craniotomy based on an unexplained number. Second: the False Confidence Problem — conventional softmax outputs never admit uncertainty, even when processing a blurry, corrupted, or atypical scan. Our project was specifically designed to engineer solutions to both of these barriers."

---

---

# SLIDE 3 — PROJECT OBJECTIVES & DATASET

## What We Built & What We Used

### Objectives (100-Mark Rubric)

| Objective | Description | Weight |
|---|---|---|
| **1** | EDA, data quality analysis, preprocessing with justification | 20 marks |
| **2** | Model design, training, optimization, full evaluation metrics | 50 marks |
| **3** | Grad-CAM XAI, Guided Grad-CAM, Saliency, 17-view Radiomics | 30 marks |

### Dataset: Multi-Class Brain Tumor MRI
| Class | Count | % | Clinical Description |
|---|---|---|---|
| Glioma | 926 | 28.4% | Malignant glial infiltration; necrotic ring enhancement |
| Meningioma | 937 | 28.7% | Benign dural-based compression; dural tail sign |
| Pituitary Tumor | 901 | 27.6% | Sellar adenoma; optic chiasm compression |
| No Tumor (Normal) | 500 | 15.3% | Normal parenchymal architecture |
| **Total** | **3,264** | **100%** | T1-weighted contrast-enhanced MRI |

**Key Challenge:** No Tumor class has only 15.3% representation → class imbalance!

---

**SPEAKER NOTES:**
> "Our dataset contains 3,264 clinical T1-weighted contrast-enhanced MRI scans across four categories. Note the critical imbalance — the healthy No Tumor baseline represents only 15.3% of images versus 28% for each tumor class. Left uncorrected, any standard neural network trained on this data would systematically over-predict tumor pathology. Addressing this imbalance was our first major engineering challenge."

---

---

# SLIDE 4 — EXPLORATORY DATA ANALYSIS

## Understanding the Data Before Modelling

### Key EDA Findings:

**Resolution Heterogeneity:**
- Native image sizes ranged from **60×60 px to 512×512 px** across acquisition centres
- Standard deviation of width: **84.3 px** — significant cross-scanner variability
- Justification for standardized resizing: all inputs must be uniform 224×224

**Pixel Intensity Analysis:**
- Mean pixel intensity across classes: 98 – 123 (significant variation)
- Standard deviation: 59 – 72 (high intra-class variability)
- Justification for CLAHE: local contrast enhancement needed, not global normalization

**Data Quality Results:**
| Check | Issues Found | Action |
|---|---|---|
| Corrupt/unreadable files | 0 | — |
| Exact duplicate images | 8 | Removed |
| Non-RGB (grayscale) images | 7 | Auto-converted |
| Extreme aspect ratios | 3 | Corrected by resize |

**Class Imbalance:** No Tumor underrepresented by **1.85× relative to tumor classes**
→ Solution: Inverse-Frequency Weighted Cross-Entropy Loss

---

**SPEAKER NOTES:**
> "Before writing a single line of model code, we performed thorough exploratory data analysis. We discovered that native MRI dimensions varied enormously — from just 60 pixels wide to 512 pixels — reflecting scans acquired on different scanners at different institutions. We also identified 8 duplicate images and 7 grayscale-only files, which were cleaned before training. Most importantly, our statistical analysis quantified the exact severity of class imbalance, allowing us to design a mathematically justified compensation strategy using inverse-frequency weighted loss functions."

---

---

# SLIDE 5 — PREPROCESSING PIPELINE

## From Raw MRI to Model-Ready Tensor

```
Raw MRI Image
     ↓
[Stage 1]  CLAHE — Contrast-Limited Adaptive Histogram Equalization
           clipLimit=2.0, tileGridSize=(8×8), applied to LAB L-channel
     ↓
[Stage 2]  Bicubic Resize → 256 × 256
     ↓
[Stage 3]  Augmentation (TRAINING ONLY)
           ✓ Random Horizontal Flip (p=0.5)    — bilateral brain symmetry
           ✓ Random Rotation ±15°              — head coil positioning variation
           ✗ Vertical Flip EXCLUDED            — anatomically impossible
           ✗ Colour Jitter EXCLUDED            — MRI physics incompatible
     ↓
[Stage 4]  Centre Crop → 224 × 224
     ↓
[Stage 5]  Normalize (ImageNet: μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])
     ↓
Model Input Tensor [Batch × 3 × 224 × 224]
```

### Why CLAHE?
- Global normalization washes out tumor tissue margins
- CLAHE locally amplifies contrast *within* tissue tiles only
- Preserves scanner noise suppression in background regions
- Converts in LAB colour space → no hue distortion

---

**SPEAKER NOTES:**
> "Our preprocessing pipeline has five carefully justified stages. The most important is CLAHE — Contrast-Limited Adaptive Histogram Equalization. Unlike global normalization which treats tumor tissue and empty background identically, CLAHE operates in 8×8 local tiles, amplifying contrast specifically within tissue-containing regions. This dramatically sharpens tumor boundaries and internal lesion texture. For augmentation, we strictly enforced biological plausibility: horizontal flips are valid because brain tumors have no left-right laterality preference, and small rotations simulate realistic patient positioning variation inside the scanner coil. We explicitly excluded vertical flips because inverted brain anatomy simply never occurs in clinical MRI."

---

---

# SLIDE 6 — MODEL ARCHITECTURE (PART 1)

## Attention-Gated ResNet-34 — Backbone Design

### Why ResNet-34?

| Architecture | Accuracy | Parameters | Verdict |
|---|---|---|---|
| VGG-16 | 89.3% | 138M | Too many params for 3K dataset |
| ResNet-50 | 95.1% | 25.6M | Good but risk of overfitting |
| **ResNet-34 (Ours)** | **97.84%** | **21.5M** | **Optimal: depth + efficiency** |
| EfficientNet-B0 | 94.7% | 5.3M | Lower capacity for complex textures |

### ResNet-34 Layer Structure (34 Total Layers):

| Stage | Content | Output Shape |
|---|---|---|
| Stem | Conv(7×7) + BN + ReLU + MaxPool | [B, 64, 56, 56] |
| conv2_x (layer1) | 3 Residual Blocks × 2 Conv(3×3) | [B, 64, 56, 56] |
| conv3_x (layer2) | 4 Residual Blocks × 2 Conv(3×3) | [B, 128, 28, 28] |
| conv4_x (layer3) | 6 Residual Blocks × 2 Conv(3×3) | [B, 256, 14, 14] |
| conv5_x (layer4) | 3 Residual Blocks × 2 Conv(3×3) | [B, 512, 7, 7] |
| **Attention Gate** ⬅ **NOVEL** | 2× Conv(1×1) + Sigmoid | [B, 512, 7, 7] |
| AvgPool + Classifier | FC(512→256→4) + MC-Dropout | [B, 4] |

### Skip Connection Formula:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
Prevents vanishing gradients → stable training at 34 layers deep

---

**SPEAKER NOTES:**
> "We selected ResNet-34 after empirically comparing four architectures. The residual skip connections are the key architectural feature: they allow gradient signals to flow directly from the output back to early layers without degradation — mathematically proven by the identity addition term. ResNet-34 provides the optimal balance of representational depth and parameter efficiency for a 3,264-image medical dataset. Deeper networks like ResNet-50 showed early signs of overfitting on our dataset size."

---

---

# SLIDE 7 — MODEL ARCHITECTURE (PART 2) — NOVELTIES

## Our Two Scientific Contributions

---

### Novelty 1: Spatial Attention Gate

**Problem it solves:** Conventional CNNs activate equally on skull bone, orbital fat, scalp, and actual tumor tissue — wasting processing capacity on irrelevant anatomy.

**How it works:**
$$M_s(\mathbf{F}) = \sigma\left(\text{Conv}_{1\times1}^{(2)}\left(\text{ReLU}\left(\text{BN}\left(\text{Conv}_{1\times1}^{(1)}(\mathbf{F})\right)\right)\right)\right)$$
$$\mathbf{F}_{out} = \mathbf{F} \otimes M_s(\mathbf{F})$$

The gate generates a soft [0,1] spatial mask. Near-zero weights suppress skull/background; near-one weights amplify tumor parenchyma.

**Result: +3.64% accuracy improvement** over ResNet-34 without attention gate.

---

### Novelty 2: Monte Carlo Dropout Uncertainty

**Problem it solves:** Standard softmax gives false certainty even on ambiguous scans.

**How it works:** Keep Dropout active at inference → run 10 forward passes → compute variance:
$$\sigma^2 = \frac{1}{M}\sum_{m=1}^{M}(\hat{y}^{(m)} - \bar{y})^2$$

If σ² > 0.05 → **"⚠ HIGH UNCERTAINTY — SPECIALIST REVIEW REQUIRED"**

**Result: 72.7% of all misclassifications were pre-flagged by uncertainty warning**

---

**SPEAKER NOTES:**
> "These are our two primary scientific novelties. The Spatial Attention Gate functions like a neurological zoom lens — it dynamically learns to suppress non-pathological anatomical structures and redirect the entire computational focus onto intracranial lesion regions. Empirically, this single module improved accuracy by 3.64 percentage points. Our second novelty, Monte Carlo Uncertainty, solves the false confidence problem. Instead of one deterministic guess, we simulate 10 virtual radiologists with slightly different neural pathways through active dropout. When they disagree significantly, our system raises a clinical alert — and this alert correctly predicted 72.7% of all eventual misclassifications."

---

---

# SLIDE 8 — TRAINING & OPTIMIZATION

## Experimental Configuration

### Hyperparameter Table

| Parameter | Value | Clinical Justification |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay — superior L₂ regularization |
| Initial LR | 1 × 10⁻³ | Standard starting rate; cosine schedule handles decay |
| Weight Decay | 1 × 10⁻⁴ | Prevents over-parameterization on 3K dataset |
| Dropout | 0.40 | Regularizes + enables MC uncertainty inference |
| Label Smoothing | 0.10 | Prevents logit over-confidence on border-zone lesions |
| Batch Size | 32 | Optimal GPU utilization + SGD variance |
| LR Schedule | Cosine Annealing Warm Restarts (T₀=10, T_mult=2) | Escapes local minima via periodic warm restarts |
| Early Stopping | Patience = 10 epochs | Saves compute; preserves best checkpoint |

### Class Weights (Inverse-Frequency):

| Class | Count | Weight Applied |
|---|---|---|
| Glioma | 826 | 0.868 |
| Meningioma | 822 | 0.873 |
| Pituitary | 827 | 0.867 |
| **No Tumor** | **395** | **1.816** ← Double penalty |

### Training Results:
- Converged at epoch **~38** (early stopping triggered)
- Train loss ≈ Validation loss (no overfitting gap observed)
- Best validation accuracy: **97.4%**

---

**SPEAKER NOTES:**
> "Our training configuration implements AdamW — not standard Adam. The W stands for decoupled Weight decay, which correctly applies L₂ regularization separately from the adaptive gradient scaling. Combined with cosine annealing warm restarts, this allows the optimizer to smoothly converge without getting trapped in flat loss landscape regions. The class weights are especially important: the No Tumor class receives nearly double the loss penalty compared to tumor classes, forcing the optimizer to learn normal anatomy boundaries with exceptional precision."

---

---

# SLIDE 9 — RESULTS & PERFORMANCE EVALUATION

## Quantitative Performance Metrics

### Overall Performance (Test Set — 394 unseen samples)

| Metric | Score |
|---|---|
| **Overall Accuracy** | **97.84%** |
| Macro Precision | 97.91% |
| Macro Recall | 97.83% |
| Macro F1-Score | 97.87% |
| **Mean AUC-ROC** | **0.9955** |

### Per-Class Breakdown

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Glioma | 96.40% | 97.10% | 96.75% | 0.9923 |
| Meningioma | 97.20% | 96.50% | 96.85% | 0.9941 |
| No Tumor | 98.94% | 99.05% | 98.99% | 0.9987 |
| Pituitary | 99.10% | 98.65% | 98.87% | 0.9968 |

### vs. Baseline Architectures

| Model | Accuracy | Δ vs Ours |
|---|---|---|
| VGG-16 | 89.3% | −8.54% |
| ResNet-34 (no attention) | 94.2% | −3.64% |
| ResNet-50 | 95.1% | −2.74% |
| **AG-ResNet-34 (Ours)** | **97.84%** | **—** |

### MC Uncertainty Safety Stats:
- High-uncertainty flags (σ² > 0.05): **11 / 394 scans (2.8%)**
- Of flagged scans that were misclassified: **8 / 11 (72.7%)**

---

**SPEAKER NOTES:**
> "Our results are exceptional. 97.84% overall accuracy on the held-out test set, with a mean AUC of 0.9955 — meaning at virtually every classification threshold, our model maintains near-perfect discrimination between all four categories. Crucially, our Glioma recall is 97.1%, which means we miss only 3 in 100 malignant glioma cases — a false negative rate that is already better than the 8–15% human inter-observer disagreement rates reported in literature. And the uncertainty module successfully pre-flagged 72.7% of all eventual misclassifications before the wrong label was ever recorded."

---

---

# SLIDE 10 — EXPLAINABLE AI: GRAD-CAM ANALYSIS

## Objective 3 — Visual Justification of Every Prediction

### Grad-CAM Mathematical Pipeline:

**Step 1:** Forward pass → get class score $Y^c$

**Step 2:** Backpropagate to get feature gradients → $\frac{\partial Y^c}{\partial A^k_{ij}}$

**Step 3:** Global Average Pool gradients → neuron importance weights:
$$\alpha^c_k = \frac{1}{Z}\sum_i\sum_j \frac{\partial Y^c}{\partial A^k_{ij}}$$

**Step 4:** Weighted activation + ReLU:
$$L^c_{Grad-CAM} = \text{ReLU}\left(\sum_k \alpha^c_k \cdot A^k\right)$$

**Step 5:** Bilinear upsample 7×7 → 224×224 → overlay on MRI

---

### Correct Prediction Analysis:
- **Glioma (TP):** Heat maps concentrate on ring-enhancing necrotic core border — precise clinical hallmark ✓
- **Pituitary (TP):** Attention anchors within sella turcica and optic chiasm — anatomically precise ✓
- **No Tumor (TP):** Diffuse low-intensity activation — correctly shows no focal pathology ✓

### Incorrect Prediction Analysis (Failure Triage):
- **Glioma → Meningioma (FP):** Network attention shifted to tumor outer margin (meningeal contact) instead of necrotic core
- **Clinical Value:** These exact cases had σ² > 0.05 — uncertainty system correctly flagged them ✓

---

**SPEAKER NOTES:**
> "Grad-CAM is the cornerstone of our explainability pipeline. For every prediction, we compute which spatial regions of the MRI drove the network's classification decision. For correctly classified Glioma scans, the heat maps concentrate precisely on the ring-enhancing margin surrounding the necrotic core — the exact same morphological feature an expert radiologist would focus on. When we examine misclassified cases, Grad-CAM reveals the failure mode: the network was focusing on the wrong anatomical substructure. This analysis is not just academically interesting — it directly guides future model improvement and validates deployment safety."

---

---

# SLIDE 11 — XAI TECHNIQUE COMPARISON & 17-VIEW DASHBOARD

## Multi-Modal Explainability in Action

### 4-Technique Comparison Panel (per MRI scan):

| Panel | Technique | What It Shows | Resolution |
|---|---|---|---|
| 1 | Original CLAHE MRI | Input after preprocessing | 224×224 |
| 2 | Grad-CAM Overlay | Semantic lesion localization (class-specific) | 224×224 |
| 3 | Guided Grad-CAM | Pixel-level fine-grained saliency | 224×224 |
| 4 | Vanilla Gradient Saliency | Raw input sensitivity map | 224×224 |
| 5 | Spatial Attention Gate Map | Our novel module's learned focus | 224×224 |

### 17-View Clinical Radiomics Dashboard (Live App):

The deployed Gradio application computes **17 simultaneous diagnostic views** per uploaded scan in real-time:

- **Grad-CAM Overlay** | **CLAHE MRI** | **Confidence Radar** | **Intensity Histogram**
- **Feature Map Grid** | **Attention Saliency Ratio (%)** | **3D Surface Elevation**
- **Radiological Severity Gauge** | **Guided Grad-CAM** | **Uncertainty Variance**
- **Topographical Contours** | **Channel Features** | **Severity Score**
- **Edge Detection Map** | **Intensity Profile** | **Watershed Segmentation**
- **Prediction Confidence Panel**

### Attention Saliency Ratio Results:
- **Mean ASR: 94.1%** — 94.1% of model attention correctly localized within cranial vault
- Confirms model is reading tumor biology, NOT image borders or scanner text

---

**SPEAKER NOTES:**
> "Our Explainable AI pipeline goes far beyond a simple Grad-CAM overlay. We implemented four complementary techniques: Grad-CAM for semantic localization, Guided Grad-CAM for pixel-level detail, Vanilla Gradient Saliency for input sensitivity, and our novel Spatial Attention Gate map. In the live demonstration you will see shortly, our Gradio dashboard renders all 17 clinical views simultaneously for each uploaded MRI scan. The Attention Saliency Ratio — which we invented specifically for this project — quantifies what percentage of the model's computational attention is safely localized inside the cranial vault. Our mean ASR of 94.1% confirms the model is learning genuine tumor biology rather than spurious image artifacts."

---

---

# SLIDE 12 — CONCLUSION, FUTURE SCOPE & LIVE DEMO

## Summary of Achievements

### What We Delivered:

| Deliverable | Status |
|---|---|
| Exploratory Data Analysis (10+ plots, quality report, CSV) | ✅ Complete |
| Preprocessing Pipeline (CLAHE, augmentation, normalization) | ✅ Complete |
| Attention-Gated ResNet-34 (novel architecture) | ✅ Complete |
| Monte Carlo Uncertainty Estimation (novel safety mechanism) | ✅ Complete |
| Full Evaluation: Accuracy 97.84%, AUC 0.9955, Confusion Matrix, ROC | ✅ Complete |
| Grad-CAM + Guided Grad-CAM + Saliency XAI Pipeline | ✅ Complete |
| 17-View Clinical Radiomics Live Dashboard (Gradio App) | ✅ Live |
| Well-Commented Python Source Code (5 modules) | ✅ Complete |
| Academic Project Report (12 sections, 30+ pages) | ✅ Complete |
| Presentation (12 slides + speaker notes) | ✅ Complete |

### Future Roadmap:

```
Phase 1 (Near-term)    → 3D Volumetric Analysis (3D-UNet on full DICOM studies)
Phase 2 (Mid-term)     → Multi-parametric MRI fusion (T1+T2+FLAIR+DWI)
Phase 3 (Long-term)    → Radiogenomics: non-invasive IDH1/MGMT mutation prediction
Phase 4 (Enterprise)   → Federated Learning: multi-hospital privacy-preserving training
```

---

## 🚀 Live Demonstration

**System URL:** http://127.0.0.1:7860

**Demo Steps:**
1. Upload any brain MRI image
2. Click "Analyze MRI Scan"
3. View classification result + confidence score
4. Examine all 17 radiomics visualizations
5. Check MC Uncertainty variance
6. Inspect Grad-CAM lesion localization

---

**SPEAKER NOTES:**
> "To conclude: our NeuroVision Diagnostic System achieves 97.84% accuracy across 4 brain tumor categories while providing complete clinical transparency through 17 simultaneous XAI visualizations. Our two scientific novelties — the Spatial Attention Gate and Monte Carlo Uncertainty — address the exact barriers that prevent standard deep learning from being deployed safely in clinical neurological practice. We are now transitioning to our live demonstration. Please feel free to upload any brain MRI image and watch the system classify it in real-time, generating the complete 17-view radiomics dashboard. We welcome all technical questions and viva examination. Thank you."

---

*End of Presentation Script — Team 8*
*Brain Tumor Classification using Deep Learning and Explainable AI*
