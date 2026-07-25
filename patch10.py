import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add percentage text directly to the cam_image (AG-ResNet34 Attention Gate)
cam_text_code = """
      # Overlay Saliency Ratio directly onto the CAM image
      cv2.putText(cam_image, f"Attention Saliency Ratio: {val:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
      
      fig_saliency_ratio, ax_sr = plt.subplots(figsize=(5, 3.5))"""
content = content.replace("fig_saliency_ratio, ax_sr = plt.subplots(figsize=(5, 3.5))", cam_text_code)

# 2. Fix the Flowchart to be perfectly aligned with no gaps (single column)
start_idx = content.find('with gr.TabItem("Technical Architecture Diagram")')
if start_idx != -1:
    end_idx = content.find('""", elem_classes="output-markdown")', start_idx) + len('""", elem_classes="output-markdown")')
    
    new_arch = '''with gr.TabItem("Technical Architecture Diagram"):
            gr.Markdown("""
            ### 🛠️ Advanced Tumor CNN Pipeline
            
            Below is the full technical flowchart mapping out the lifecycle of an MRI scan passing through our **Attention-Gated ResNet-34** architecture, combined with **Monte Carlo Uncertainty Estimation** and **Explainable Radiomics**.
            
            ```mermaid
            %%{init: {'theme': 'dark', 'themeVariables': { 'lineColor': '#94a3b8', 'textColor': '#f8fafc' }}}%%
            graph TD
                classDef input fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:white,font-weight:bold,padding:10px
                classDef process fill:#10b981,stroke:#047857,stroke-width:2px,color:white,font-weight:bold,padding:10px
                classDef attention fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:white,font-weight:bold,padding:10px
                classDef output fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:white,font-weight:bold,padding:10px
                classDef mc fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:white,font-weight:bold,padding:10px

                A["<b>Raw Patient MRI Scan</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 512x512x3 • Feat: RGB Pixels<br/>Importance: Preserves original unaltered anatomy"]:::input
                
                A --> B["<b>CLAHE Preprocessing & Normalization</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 512x512x3 • Feat: Contrast-Limited<br/>Importance: Maximizes tissue separability"]:::process
                
                B --> C["<b>ResNet-34 Feature Extractor Backbone</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 16x16x512 • Feat: High-Level Semantics<br/>Importance: Extracts hierarchical patterns"]:::process
                
                subgraph Spatial Attention Gate
                C --> D["<b>1x1 Conv Bottleneck</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 16x16x1 • Feat: Spatial Logits<br/>Importance: Compresses channel dimensions"]:::attention
                
                D --> E["<b>Sigmoid Activation Function</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 16x16x1 • Feat: Probabilities 0 to 1<br/>Importance: Generates non-linear masking"]:::attention
                
                E --> F["<b>2D Attention Saliency Map</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 16x16x1 • Feat: Weighted ROI<br/>Importance: Focuses network on tumor regions"]:::attention
                end
                
                F --> G["<b>Global Average Pooling</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 1x512 • Feat: Flattened Vector<br/>Importance: Spatial translation invariance"]:::process
                
                subgraph Monte Carlo Uncertainty Estimation
                G --> H["<b>Dropout Layer 1 p=0.4</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 1x512 • Feat: Stochastic Mask<br/>Importance: Introduces initial model variance"]:::mc
                
                H --> I["<b>Fully Connected Layer 256</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 1x256 • Feat: Dense Features<br/>Importance: High-level reasoning"]:::process
                
                I --> J["<b>Dropout Layer 2 p=0.4</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 1x256 • Feat: Stochastic Mask<br/>Importance: Prevents overconfidence"]:::mc
                
                J --> M["<b>10x Stochastic Forward Passes</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 10x4 • Feat: Softmax Ensemble<br/>Importance: Simulates Bayesian approximation"]:::mc
                end
                
                M --> K["<b>Softmax Classification Ensemble</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 1x4 • Feat: Mean Probabilities<br/>Importance: Final robust classification"]:::output
                
                K --> L("<b>Final Tumor Diagnosis + Variance Confidence</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Feat: Human-Readable Diagnosis<br/>Importance: Clinical decision support"):::output
                
                F --> N("<b>Visual Extracted Heatmap UI</b><br/><hr style='border: 1px dashed #cbd5e1; margin:5px;'/>☁️ Dim: 512x512 • Feat: Interpolated Heatmap<br/>Importance: Explainability & Doctor Trust"):::output
            ```
            
            ---
            
            ### 🔬 Diagnostic Classification Output
            The final step of the pipeline maps the extracted mathematical features into **4 distinct clinical categories**. Here is how the system identifies each:
            
            - **1️⃣ Glioma Tumor**: Often highly aggressive and originating from glial cells. The AI specifically looks for invasive, irregular borders and diffuse/chaotic textures within the brain mass.
            - **2️⃣ Meningioma Tumor**: Usually slow-growing and benign, forming on the membranes covering the brain. The AI looks for well-defined, pushing borders and homogenous density on the outer edges of the brain.
            - **3️⃣ Pituitary Tumor**: Located at the base of the brain (the pituitary gland). The AI leverages strong spatial attention priors (center-bottom localization) to pinpoint these specific abnormalities.
            - **4️⃣ No Tumor (Healthy)**: Normal baseline parenchyma with healthy structural symmetry and no hyper-intense mass regions.
            
            #### 🧠 How the System Predicts
            To maximize medical safety, the system does not just guess once. It executes **10 separate stochastic forward passes** through the pipeline using Monte Carlo Dropout. This simulates having a panel of 10 different virtual radiologists review the exact same scan. 
            
            The system mathematically averages these 10 probability vectors together to form the **Softmax Classification Ensemble**. The category with the highest final average probability becomes the **Final Predicted Diagnosis**. If the 10 virtual passes heavily disagree with each other, the system automatically spikes the **Uncertainty Variance** graph, alerting the doctor that the scan is highly ambiguous and requires immediate human oversight.
            """, elem_classes="output-markdown")'''
    
    content = content[:start_idx] + new_arch + content[end_idx:]
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully completely unified the flowchart for perfect alignment!')
