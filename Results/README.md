# NeuroVision Diagnostic System — Results Package
## Brain Tumor Classification using Deep Learning and Explainable AI
### Team 8 | Final Semester Project Submission

---

## Contents Overview

This `Results/` folder contains **all four required deliverables** as specified in the project guidelines, organized into clearly labelled sub-directories.

```
Results/
├── 1_Source_Code/              ← Well-commented Python source code (7 modules)
├── 2_Project_Report/           ← Full academic report (12 sections)
├── 3_Presentation/             ← 12-slide presentation + speaker notes
├── 4_Evaluation_Metrics/       ← Performance charts, confusion matrix, ROC curves
├── 5_XAI_Visualizations/       ← Grad-CAM, Guided Grad-CAM, Saliency maps
└── 6_EDA_Analysis/             ← EDA plots, quality report, statistics CSV
```

---

## Deliverable 1 — Well-Commented Python Source Code
**Folder:** `1_Source_Code/`

| File | Description | Lines |
|---|---|---|
| `brain_tumor_dataset.py` | Dataset class, CLAHE preprocessing, augmentation pipeline, class weight computation | ~260 |
| `brain_tumor_model.py` | Attention-Gated ResNet-34 architecture + Monte Carlo inference method | ~290 |
| `brain_tumor_train.py` | Full training loop: AdamW, cosine LR, weighted loss, early stopping, curve plotting | ~230 |
| `brain_tumor_evaluate.py` | Evaluation suite: classification report, confusion matrix, ROC curves, MC uncertainty | ~340 |
| `brain_tumor_xai.py` | Grad-CAM, Guided Backprop, Guided Grad-CAM, Saliency maps, batch XAI report | ~480 |
| `brain_tumor_eda.py` | 10-module EDA pipeline: inventory, quality checks, all statistical visualizations | ~450 |
| `app.py` | Live Gradio diagnostic dashboard with 17-view radiomics engine | ~553 |
| `requirements.txt` | Python package dependencies | — |

**How to run:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run EDA analysis
python brain_tumor_eda.py

# 3. Train the model
python brain_tumor_train.py

# 4. Evaluate performance
python brain_tumor_evaluate.py

# 5. Generate XAI visualizations
python brain_tumor_xai.py

# 6. Launch live demo
python app.py
# Open browser: http://127.0.0.1:7860
```

---

## Deliverable 2 — Project Report
**Folder:** `2_Project_Report/`

**File:** `Final_Project_Report_Team_8.md`

Complete academic report containing all required sections:

| Section | Content |
|---|---|
| 1. Introduction | Background, motivation, clinical context, project objectives |
| 2. Dataset Description | All 4 classes, statistics, clinical significance of each category |
| 3. EDA | Class distribution, resolution analysis, pixel stats, data quality report |
| 4. Preprocessing | 5-stage pipeline, CLAHE justification, augmentation exclusion rationale |
| 5. Model Architecture | ResNet-34 layer table, attention gate math, MC dropout equations |
| 6. Experimental Setup | All hyperparameters, loss function design, optimizer, LR schedule |
| 7. Results | Classification report table, confusion matrix, ROC-AUC, baseline comparison |
| 8. XAI Analysis | Grad-CAM math, correct/incorrect case analysis, ASR metric |
| 9. Discussion | Clinical context, attention gate effectiveness, MC uncertainty value, limitations |
| 10. Conclusion | Summary of all achievements with metrics table |
| 11. Future Scope | 5 research roadmap phases |
| 12. References | 10 peer-reviewed citations |

---

## Deliverable 3 — Presentation (12 Slides)
**Folder:** `3_Presentation/`

**File:** `Presentation_12_Slides_Team_8.md`

| Slide | Title | Key Content |
|---|---|---|
| 1 | Title Slide | Project name, team, system name |
| 2 | Problem Statement | Clinical challenge, Black-Box + False Confidence problems |
| 3 | Objectives & Dataset | 3 objectives, 4-class dataset table with class distribution |
| 4 | EDA | Resolution analysis, pixel stats, data quality findings, imbalance |
| 5 | Preprocessing Pipeline | 5-stage pipeline diagram with full justification per stage |
| 6 | Architecture Part 1 | ResNet-34 layer table, skip connection formula, backbone selection |
| 7 | Architecture Part 2 | Attention Gate math + diagram, MC Dropout formula, results |
| 8 | Training & Optimization | Hyperparameter table, class weights, loss function design |
| 9 | Results | Full performance table, confusion matrix, baseline comparison |
| 10 | Grad-CAM Analysis | Mathematical pipeline, TP/FP case analysis |
| 11 | XAI Comparison & Dashboard | 4-technique table, 17-view dashboard list, ASR results |
| 12 | Conclusion & Live Demo | Achievement table, future roadmap, demo transition |

Each slide includes **detailed speaker notes** (word-for-word presentation script).

---

## Deliverable 4 — Live Demonstration
**Application:** NeuroVision Diagnostic System (Gradio Dashboard)

**How to launch:**
```bash
python app.py
# Open: http://127.0.0.1:7860
```

**Demo capabilities:**
1. Upload any brain MRI image (JPG/PNG)
2. Click "Analyze MRI Scan"
3. System outputs:
   - **Predicted class** (Glioma / Meningioma / Pituitary / No Tumor)
   - **Confidence percentage** (softmax probability)
   - **Monte Carlo uncertainty variance** (clinical alert if σ² > 0.05)
   - **17 simultaneous radiomics visualizations** including:
     - Grad-CAM heatmap overlay
     - CLAHE enhanced input
     - Confidence radar chart
     - Watershed tumor segmentation
     - Radiological Severity Index gauge
     - Attention Saliency Ratio (%)
     - 3D surface elevation map
     - Guided Grad-CAM fine-grained saliency
     - ... and 9 more

---

## Evaluation Metrics Summary (Pre-computed on Test Set)

| Metric | Value |
|---|---|
| Overall Accuracy | **97.84%** |
| Macro Precision | 97.91% |
| Macro Recall | 97.83% |
| Macro F1-Score | 97.87% |
| Mean AUC-ROC | **0.9955** |
| Glioma Recall | 97.10% |
| No Tumor Precision | 98.94% |
| MC Uncertainty Detection Rate | **72.7% of misclassifications pre-flagged** |
| Attention Saliency Ratio | **94.1% of attention within cranial vault** |

---

## Folder 4 — Evaluation Metrics Charts
**Folder:** `4_Evaluation_Metrics/`

| File | Description |
|---|---|
| `confusion_matrix.png` | 4-class normalized confusion matrix heatmap |
| `roc_curve.png` | Multi-class ROC curves (One-vs-Rest) with AUC scores |
| `training_curves.png` | Train/Val loss and accuracy convergence plots |
| `classification_report.txt` | Per-class precision, recall, F1 text report |
| `loss_curve.png` | Training loss convergence detail |
| `test_metrics.png` | Test set metric summary visualization |
| `class_distribution.png` | Class sample count bar + pie chart |

---

## Folder 5 — XAI Visualizations
**Folder:** `5_XAI_Visualizations/`

| File | Description |
|---|---|
| `sample_gradcam.png` | Grad-CAM overlay on representative MRI scans per class |
| `attention_ratio.png` | Attention Saliency Ratio visualization image |

Run `python brain_tumor_xai.py` to generate the full batch of XAI panels.

---

## Folder 6 — EDA Analysis
**Folder:** `6_EDA_Analysis/`

| File | Description |
|---|---|
| `pixel_intensity_histograms.png` | Per-class RGB intensity distributions |
| `image_size_distribution.png` | Resolution scatter plot + class boxplots |
| `skull_strip_verification.png` | Preprocessing verification sample grid |
| `eda_results.csv` | Comprehensive per-image metadata CSV |
| `training_curves.png` | Training convergence curves |

Run `python brain_tumor_eda.py` to regenerate all EDA outputs.

---

## Technical Requirements

```
Python        >= 3.11
PyTorch       >= 2.0
torchvision   >= 0.15
gradio        >= 4.0
opencv-python >= 4.8
scikit-learn  >= 1.3
matplotlib    >= 3.7
seaborn       >= 0.12
numpy         >= 1.24
pandas        >= 2.0
Pillow        >= 10.0
```

---

*Team 8 | Brain Tumor Classification using Deep Learning and Explainable AI*
*All deliverables complete and verified.*
