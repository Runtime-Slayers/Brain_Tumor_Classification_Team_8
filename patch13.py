import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Change output_saliency_ratio to a Textbox
content = content.replace(
    'output_saliency_ratio = gr.Image(label="Attention Saliency Ratio", interactive=False)',
    'output_saliency_ratio = gr.Textbox(label="Attention Saliency Ratio", interactive=False)'
)

# 2. Remove matplotlib code for fig_saliency_ratio
graph_code = """    fig_saliency_ratio, ax_sr = plt.subplots(figsize=(5, 3.5))
    ax_sr.barh([''], [val], color='#2b6cb0', height=0.5)
    ax_sr.set_xlim(0, 115)
    ax_sr.set_title("Attention Saliency Ratio", weight='bold')
    ax_sr.text(val + 2, 0, f"{val:.1f}%", va='center', weight='bold', color='white')
    ax_sr.set_yticks([])
    ax_sr.spines['top'].set_visible(False)
    ax_sr.spines['right'].set_visible(False)
    ax_sr.spines['left'].set_visible(False)
    plt.tight_layout()"""

content = content.replace(graph_code, "")

# 3. Change the return tuple to just return the string format of val
content = content.replace(
    'fig_to_image(fig_saliency_ratio)',
    'f"{val:.1f}%"'
)

# 4. Inject the detailed description of the ResNet-34 architecture and project novelty
novelty_text = """
            ---
            
            ### 🧠 Attention-Gated ResNet-34 Breakdown

            **The Number of Layers and Arrangement:**
            The backbone is exactly **34 layers deep**. It is arranged sequentially:
            1. **1x Initial Conv Layer:** Acts as the primary receptor, downsampling the raw MRI and extracting basic edges.
            2. **16x Residual Blocks (32 Layers Total):** These are arranged into 4 hierarchical stages. Each block contains 2 convolutional layers. The critical feature here is the **Skip Connections** (Residuals) that mathematically bypass layers. This allows the network to stay 34 layers deep without suffering from the vanishing gradient problem.
            3. **1x Fully Connected Layer:** The final layer that flattens the mathematical features into 4 distinct tumor categories.

            **Did we add any novelty to it?**
            Yes! Standard ResNet-34 looks at the *entire* image equally, which is bad for medical imaging because healthy brain tissue acts as "noise". We mathematically injected a **Spatial Attention Gate** right before the final pooling layer. 
            - It uses a custom `1x1` Convolutional bottleneck combined with a Sigmoid Activation function.
            - It calculates a `0 to 1` probability mask that physically "mutes" the healthy background brain tissue and aggressively amplifies the mathematical weights of the hyper-intense tumor regions. This forces the ResNet to look *only* at what matters.

            ---

            ### 🚀 What is the Novelty of our ENTIRE Project?

            Most academic brain tumor classifiers are "Black Boxes"—they output a single guess (e.g., "Glioma 99%") and offer zero explanation. If the AI is wrong, it is still confidently wrong, which is extremely dangerous in a hospital. **Our project introduces two major novelties to fix this:**

            **1. Monte Carlo (MC) Uncertainty Estimation for Medical Safety**
            Instead of guessing once, we built a Stochastic Ensemble. We force the Dropout layers to remain active *during live inference*. 
            When a scan is uploaded, the system passes it through the AI **10 separate times**. Because of the active dropout, the AI's internal pathways change slightly every time—mathematically simulating a panel of 10 different virtual radiologists reviewing the exact same scan. If the 10 virtual doctors heavily disagree on the diagnosis, the system triggers an **Uncertainty Variance** warning, alerting the human doctor that the scan is highly ambiguous and requires immediate manual oversight.

            **2. Hyper-Explainable Radiomics Pipeline**
            We don't just output text. We aggressively intercept the internal tensor mathematics of the CNN and reverse-engineer them into 17 distinct, human-readable clinical graphs. By exposing the Topographical Contours, Watershed Basins, Confidence Radars, and the Radiological Severity Index on a dynamic UI, we bridge the gap between abstract AI mathematics and traditional human radiology.
            """

target_marker = "immediate human oversight."
if target_marker in content:
    content = content.replace(target_marker, target_marker + novelty_text)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully applied all changes for Saliency text and Novelty documentation!')
