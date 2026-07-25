import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update fig_to_image to make text white for dark mode
old_fig_to_image = '''def fig_to_image(fig):
    import io
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False, facecolor='#0f172a')'''

new_fig_to_image = '''def fig_to_image(fig):
    import io
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    
    for ax in fig.axes:
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        if ax.title:
            ax.title.set_color('white')
            
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=False, facecolor='#0f172a')'''

code = code.replace(old_fig_to_image, new_fig_to_image)

# 2. Update fig_sev text color to white
old_sev_text = '''ax_sev.text(v + 2, i, f"{v:.1f}", va='center', weight='bold')'''
new_sev_text = '''ax_sev.text(v + 2, i, f"{v:.1f}", va='center', weight='bold', color='white')'''
code = code.replace(old_sev_text, new_sev_text)

# Also update the title color of fig_sev explicitly if needed, but fig_to_image handles it.
# Wait, ax_sr text color was set to black. Let's make it white.
old_sr_text = '''ax_sr.text(val + 2, 0, f"{val:.1f}%", va='center', weight='bold', color='black')'''
new_sr_text = '''ax_sr.text(val + 2, 0, f"{val:.1f}%", va='center', weight='bold', color='white')'''
code = code.replace(old_sr_text, new_sr_text)

# Ensure left margin on severity index is wide enough
old_sev_plot = '''fig_sev, ax_sev = plt.subplots(figsize=(6, 3.5))'''
new_sev_plot = '''fig_sev, ax_sev = plt.subplots(figsize=(6, 3.5))
    plt.subplots_adjust(left=0.25)'''
code = code.replace(old_sev_plot, new_sev_plot)

# 3. Replace the Technical Architecture text with Mermaid flowchart
old_arch = '''        with gr.TabItem("Technical Architecture (Explained Simply)"):
            gr.Markdown("""
            ### 🧠 How This Project Works (From Scratch, In Plain English)
            We built this AI to act like a highly trained human radiologist. It doesn't just guess; it breaks the image down step-by-step. Here is exactly how it works in plain English:

            #### Step 1: Preprocessing (Cleaning the Image)
            - **What we use**: CLAHE (Contrast Limited Adaptive Histogram Equalization).
            - **Simple Explanation**: When you take a photo in the dark, you might use a filter to brighten it. CLAHE is a medical-grade filter. It enhances the contrast of the MRI scan so the AI can clearly see the difference between healthy brain tissue and a dark/blurry tumor without washing out the image.

            #### Step 2: The Brain of the AI (AG-ResNet34)
            - **What we use**: An Attention-Gated ResNet-34 Convolutional Neural Network.
            - **Simple Explanation**: A standard AI looks at the whole brain scan equally, which is inefficient. Our AI has a mathematical "Attention" mechanism. 
              - *Spatial Attention Gate*: Tells the AI **WHERE** to look (e.g., "Ignore the skull, zoom in on the center of the brain").
            
            #### Step 3: Explainable AI (Showing Its Work)
            - **What we use**: Native Attention Maps, Saliency Maps, and Watershed Segmentation.
            - **Simple Explanation**: If a doctor says "You have a tumor," you want them to point to it! These algorithms force the AI to generate heatmaps (red/yellow colors) physically highlighting the exact pixels on the scan that caused it to make its diagnosis. **Watershed Segmentation** acts like a topographical map, mathematically drawing physical boundaries around the tumor like a lake.

            #### Step 4: Medical Safety (Knowing When It's Unsure)
            - **What we use**: Monte Carlo Dropout Uncertainty.
            - **Simple Explanation**: Standard AI is dangerously overconfident (it will say it is 100% sure even if it's completely wrong). We use "Monte Carlo Dropout" to force the AI to look at the same image 10 different times with slightly different "blindfolds" on. If it gives the exact same answer 10 times, we know it's highly confident. If it hesitates, the Uncertainty Variance graph spikes, warning the doctor to double-check the scan!
            """)'''

new_arch = '''        with gr.TabItem("Technical Architecture Diagram"):
            gr.Markdown("""
            ### 🛠️ Advanced Tumor CNN Pipeline
            
            Below is the full technical flowchart mapping out the lifecycle of an MRI scan passing through our **Attention-Gated ResNet-34** architecture, combined with **Monte Carlo Uncertainty Estimation** and **Explainable Radiomics**.
            
            ```mermaid
            graph TD
                classDef input fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:white,font-weight:bold
                classDef process fill:#10b981,stroke:#047857,stroke-width:2px,color:white,font-weight:bold
                classDef attention fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:white,font-weight:bold
                classDef output fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:white,font-weight:bold
                classDef mc fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:white,font-weight:bold
            
                A[Raw Patient MRI Scan]:::input --> B[CLAHE Preprocessing & Normalization]:::process
                B --> C[ResNet-34 Feature Extractor Backbone]:::process
                
                subgraph Spatial Attention Gate
                C --> D[1x1 Conv Bottleneck]:::attention
                D --> E[Sigmoid Activation Function]:::attention
                E --> F[2D Attention Saliency Map]:::attention
                end
                
                F --> G[Global Average Pooling]:::process
                
                subgraph Monte Carlo Uncertainty Estimation
                G --> H[Dropout Layer 1 p=0.4]:::mc
                H --> I[Fully Connected Layer 256]:::process
                I --> J[Dropout Layer 2 p=0.4]:::mc
                J --> M[10x Stochastic Forward Passes]:::mc
                end
                
                M --> K[Softmax Classification Ensemble]:::output
                K --> L(Final Tumor Diagnosis + Variance Confidence):::output
                F --> N(Visual Extracted Heatmap UI):::output
            ```
            """)'''

code = code.replace(old_arch, new_arch)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("app.py successfully updated with patch 2.")
