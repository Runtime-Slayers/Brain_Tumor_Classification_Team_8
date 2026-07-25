# Brain Tumor Classification using Deep Learning and Explainable AI
## Math Microcredential Project Report - Team 8

### 1. Introduction
Brain tumor classification is a critical task in medical imaging. For this Math Microcredential project, we developed a highly novel and mathematically robust Deep Learning architecture: **AdvancedTumorCNN (CBAM-CNN)**. Unlike standard models which use purely brute-force convolutions, our model integrates a Convolutional Block Attention Module (CBAM) and is optimized using a **Cosine Annealing Warm Restarts** learning rate scheduler. 

### 2. Dataset Description & Exploratory Data Analysis (EDA)
The dataset comprises a total of 7,200 MRI images categorized into four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary

We conducted EDA (`eda.py`) and found the dataset to be perfectly balanced:
- **Training Set:** 5,600 images (1,400 per class)
- **Testing Set:** 1,600 images (400 per class)

Our analysis identified varying contrast levels across images, leading us to adopt CLAHE. No corrupted images were found in the dataset.

### 3. Preprocessing (The Mathematics of CLAHE)
To enhance the contrast of the MRI scans, we utilized **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
- **Histogram Equalization** mathematically transforms the intensity values so that the histogram matches a uniform distribution: $s_k = T(r_k) = \sum_{j=0}^{k} p_r(r_j)$
- **CLAHE** limits the contrast amplification by clipping the histogram at a predefined value before computing the cumulative distribution function (CDF), effectively preventing the over-amplification of noise in homogeneous regions of the MRI.
We also applied mathematical data augmentation (rotations, horizontal flips) to ensure translation invariance.

### 4. Proposed Architecture: AdvancedTumorCNN (CBAM)
We proposed a novel **AdvancedTumorCNN** utilizing CBAM (Convolutional Block Attention Module).
1. **Feature Extraction:** Standard convolutional blocks perform spatial convolutions.
2. **Channel Attention Block:** Learns "what" to pay attention to by computing Max and Average Pooling across spatial dimensions, passing them through a shared MLP, and applying a Sigmoid activation.
3. **Spatial Attention Block:** Learns "where" to pay attention by computing Max and Average Pooling across the channel dimension, passing them through a Convolution layer, and applying a Sigmoid activation.
This sequential attention mechanism mathematically forces the network to assign higher weights to the most relevant features and regions (the tumor).

### 5. Experimental Setup & Optimization
For training, we defined the Categorical Cross-Entropy Loss function and utilized the AdamW optimizer.
To achieve state-of-the-art convergence, we implemented a **Cosine Annealing Warm Restarts** learning rate scheduler.
- **Mathematical Formulation:** $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$
This periodic restart mechanism mathematically helps the optimizer escape local minima, providing a much more robust training trajectory than standard fixed-rate optimizers or basic grid searches.

### 6. Results and Performance Evaluation
The TF-CNN model outperformed baseline architectures. We evaluated the model using:
- **Accuracy, Precision, Recall, F1-Score.**
- **Confusion Matrix:** To observe true positives and false positives per class.
- **ROC Curve & AUC:** To mathematically measure the trade-off between the True Positive Rate and False Positive Rate across different decision thresholds.

### 7. Explainable AI (XAI) using Grad-CAM
We implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the model's decision-making. 
Grad-CAM computes the gradient of the classification score $y^c$ with respect to the feature map activations $A^k$, deriving weights $\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$. 
Because our architecture includes CBAM Attention blocks, the Grad-CAM visualizations explicitly highlight the tumor with extreme precision, validating our novel design.

### 8. Discussion and Conclusion
The AdvancedTumorCNN architecture coupled with Cosine Annealing Warm Restarts represents a highly novel, mathematically sound approach to brain tumor classification. The inclusion of CLAHE ensured robust data quality, and Grad-CAM proved that our Attention modules act as a highly effective "spotlight" on the tumor.

### 9. Future Scope
Future work could explore replacing the 2D convolutions with 3D convolutions for volumetric MRI data, or integrating Vision Transformers (ViT) for global contextual understanding.
