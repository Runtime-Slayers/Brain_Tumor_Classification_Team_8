# Brain Tumor Classification using Deep Learning and Explainable AI
**Team 8 - Math Microcredential Project**

---
## Slide 1: Introduction
- **Objective:** Classify brain MRI scans into 4 categories (Glioma, Meningioma, Pituitary, No Tumor).
- **Motivation:** Accurate tumor detection is vital.
- **Our Novel Approach:** We designed **TumorFocus-CNN (TF-CNN)**, integrated with **CLAHE preprocessing** and optimized via **A-Star Hyperparameter Search**.

---
## Slide 2: Dataset & Preprocessing (CLAHE)
- **Dataset:** 4 classes, Training & Testing splits.
- **Issue:** MRI scans often suffer from poor contrast.
- **Solution - CLAHE (Contrast Limited Adaptive Histogram Equalization):**
  - Mathematically maps intensity values to a wider distribution without amplifying noise.
  - Improves edge visibility of tumors.

---
## Slide 3: Exploratory Data Analysis (EDA)
- Analyzed Class Distribution to check for imbalances.
- Extracted Pixel Intensity histograms.
- Corrupted images were flagged and cleaned during data loading.
- Data Augmentation applied (Rotations, Flips) for robust training.

---
## Slide 4: Proposed Architecture: TumorFocus-CNN
- Standard CNNs look at the entire image blindly.
- **Our Model:** We added a custom **Spatial Attention Module** (The "Spotlight").
- **Mathematics of Attention:** 
  - Computes Max & Average Pooling across channels.
  - Passes through Conv2D + Sigmoid to generate an Attention Map $A \in [0, 1]$.
  - Multiplies features by $A$ to highlight the tumor mathematically before classification.

---
## Slide 5: A-Star Hyperparameter Optimization
- Grid Search and Random Search are inefficient.
- **Our Novel Approach:** We used **A-Star (A*) Search**.
- **State Space:** Discretized grid of Learning Rates and Dropout values.
- **Cost $g(n)$:** Validation Loss after short proxy training.
- Result: Found optimal hyperparameters faster by expanding only the most promising nodes in the grid.

---
## Slide 6: Model Training & Evaluation
- Trained using Adam Optimizer and Categorical Cross-Entropy Loss.
- Evaluated on Test Set.
- **Metrics Computed:** Accuracy, Precision, Recall, F1-Score.
- ROC Curves and AUC plotted to show mathematical robustness across thresholds.

---
## Slide 7: Explainable AI (Grad-CAM)
- Deep learning is often a "Black Box".
- **Grad-CAM (Gradient-weighted Class Activation Mapping):** Uses the gradients of the target concept to produce a localization map.
- Because of our Spatial Attention block, our Grad-CAM explicitly highlights the exact bounding region of the tumor.

---
## Slide 8: Grad-CAM Results
- *(Insert images of correctly classified Grad-CAM here)*
- The heatmap perfectly aligns with the tumor location.
- *(Insert images of incorrectly classified Grad-CAM here)*
- Helps us understand *why* the model failed (e.g., looking at noise).

---
## Slide 9: Live Demonstration (Gradio App)
- We built a custom web application using **Gradio**.
- **Features:** 
  - Upload any MRI image.
  - Applies CLAHE in real-time.
  - Outputs prediction probabilities.
  - Renders the Grad-CAM "Spotlight" heatmap dynamically.

---
## Slide 10: Conclusion & Future Scope
- **Conclusion:** Our TumorFocus-CNN with A-Star Optimization and CLAHE achieved excellent accuracy and high explainability, proving that lightweight, attention-driven models are powerful.
- **Future Scope:** Extend A-Star optimization to full Neural Architecture Search (NAS) and adapt the model for 3D MRI volumes.
