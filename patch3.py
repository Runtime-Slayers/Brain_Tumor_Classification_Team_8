with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find('with gr.TabItem("Technical Architecture Diagram")')
if start_idx != -1:
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
                classDef cloudNote fill:#334155,stroke:#94a3b8,stroke-width:1px,color:#e2e8f0,font-size:12px,stroke-dasharray: 4 4

                A[Raw Patient MRI Scan]:::input
                A_cloud([☁️ Dim: 512x512x3 | Feat: RGB Pixels<br/>Importance: Preserves original unaltered anatomy]):::cloudNote
                A -.-> A_cloud
                
                A --> B[CLAHE Preprocessing & Normalization]:::process
                B_cloud([☁️ Dim: 512x512x3 | Feat: Contrast-Limited<br/>Importance: Maximizes tissue separability]):::cloudNote
                B -.-> B_cloud
                
                B --> C[ResNet-34 Feature Extractor Backbone]:::process
                C_cloud([☁️ Dim: 16x16x512 | Feat: High-Level Semantics<br/>Importance: Extracts hierarchical patterns]):::cloudNote
                C -.-> C_cloud
                
                subgraph Spatial Attention Gate
                C --> D[1x1 Conv Bottleneck]:::attention
                D_cloud([☁️ Dim: 16x16x1 | Feat: Spatial Logits<br/>Importance: Compresses channel dimensions]):::cloudNote
                D -.-> D_cloud
                
                D --> E[Sigmoid Activation Function]:::attention
                E_cloud([☁️ Dim: 16x16x1 | Feat: Probabilities 0 to 1<br/>Importance: Generates non-linear masking]):::cloudNote
                E -.-> E_cloud
                
                E --> F[2D Attention Saliency Map]:::attention
                F_cloud([☁️ Dim: 16x16x1 | Feat: Weighted ROI<br/>Importance: Focuses network on tumor regions]):::cloudNote
                F -.-> F_cloud
                end
                
                F --> G[Global Average Pooling]:::process
                G_cloud([☁️ Dim: 1x512 | Feat: Flattened Vector<br/>Importance: Spatial translation invariance]):::cloudNote
                G -.-> G_cloud
                
                subgraph Monte Carlo Uncertainty Estimation
                G --> H[Dropout Layer 1 p=0.4]:::mc
                H_cloud([☁️ Dim: 1x512 | Feat: Stochastic Mask<br/>Importance: Introduces initial model variance]):::cloudNote
                H -.-> H_cloud
                
                H --> I[Fully Connected Layer 256]:::process
                I_cloud([☁️ Dim: 1x256 | Feat: Dense Features<br/>Importance: High-level reasoning]):::cloudNote
                I -.-> I_cloud
                
                I --> J[Dropout Layer 2 p=0.4]:::mc
                J_cloud([☁️ Dim: 1x256 | Feat: Stochastic Mask<br/>Importance: Prevents overconfidence]):::cloudNote
                J -.-> J_cloud
                
                J --> M[10x Stochastic Forward Passes]:::mc
                M_cloud([☁️ Dim: 10x4 | Feat: Softmax Ensemble<br/>Importance: Simulates Bayesian approximation]):::cloudNote
                M -.-> M_cloud
                end
                
                M --> K[Softmax Classification Ensemble]:::output
                K_cloud([☁️ Dim: 1x4 | Feat: Mean Probabilities<br/>Importance: Final robust classification]):::cloudNote
                K -.-> K_cloud
                
                K --> L(Final Tumor Diagnosis + Variance Confidence):::output
                L_cloud([☁️ Feat: Human-Readable Diagnosis<br/>Importance: Clinical decision support]):::cloudNote
                L -.-> L_cloud
                
                F --> N(Visual Extracted Heatmap UI):::output
                N_cloud([☁️ Dim: 512x512 | Feat: Interpolated Heatmap<br/>Importance: Explainability & Doctor Trust]):::cloudNote
                N -.-> N_cloud
            ```
            """, elem_classes="output-markdown")'''
    
    content = content[:start_idx] + new_arch + content[end_idx:]
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully patched flowchart!')
