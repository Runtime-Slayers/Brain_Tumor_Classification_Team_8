import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start of the Architecture tab
start_idx = content.find('with gr.TabItem("Technical Architecture')
if start_idx != -1:
    # Find the end of this markdown block
    end_idx = content.find('""", elem_classes="output-markdown")', start_idx) + len('""", elem_classes="output-markdown")')
    
    new_arch = '''with gr.TabItem("Technical Architecture Diagram"):
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
            """, elem_classes="output-markdown")'''
    
    content = content[:start_idx] + new_arch + content[end_idx:]
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully patched flowchart!')
else:
    print('Could not find the Architecture tab.')
