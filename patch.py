import re
import io

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace all gr.Plot(label=X) with gr.Image(label=X, interactive=False, type='numpy')
code = re.sub(r'gr\.Plot\(label=(.*?)\)', r'gr.Image(label=\1, interactive=False)', code)

# Add the fig_to_image function before the predict function
helper_func = '''
def fig_to_image(fig):
    import io
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False, facecolor='#0f172a')
    buf.seek(0)
    img = Image.open(buf)
    arr = np.array(img)
    plt.close(fig)
    return arr
'''
if "def fig_to_image" not in code:
    code = code.replace('def predict(image):', helper_func + '\ndef predict(image):')

# Update the clinical report logic inside predict()
old_report = '''    clinical_report = f"""
## RADIOLOGICAL AI ANALYSIS REPORT
**DATE**: {timestamp}  
**PRIMARY AI DIAGNOSIS**: **{formatted_label}**  
**MC CONFIDENCE**: {probs[predicted_class]*100:.2f}%  

### 🔬 RADIOMICS OBSERVATIONS (EASY TO UNDERSTAND)
- **Tumor Size**: The tumor takes up approximately **{relative_size*100:.2f}%** of the visible brain area on this slice.
- **Tumor Texture**: The inside of the tumor is **{hetero_type}**. (Chaotic textures often indicate higher severity).
- **Tumor Edges**: The physical boundary of the tumor is **{border_type}**.
- **Severity Score**: Based on its size, texture, and pathology, this tumor gets a severity score of **{severity_score:.1f} out of 100**.

---
{medical_inference[predicted_label]}
"""'''

new_report = '''    if predicted_label == 'notumor':
        obs_size = "N/A (No pathological mass detected)"
        obs_texture = "Normal background parenchyma density observed."
        obs_edges = "N/A"
        obs_sev = "0.0 out of 100 (Healthy baseline)"
    else:
        size_desc = "Massive/Diffuse" if relative_size > 0.3 else "Moderate" if relative_size > 0.1 else "Focal/Localized"
        obs_size = f"The tumor takes up approximately **{relative_size*100:.1f}%** of the visible brain area ({size_desc})."
        obs_texture = f"The internal density is **{hetero_type}**."
        obs_edges = f"The physical boundary is **{border_type}**."
        obs_sev = f"Based on combined radiomics, the severity score is **{severity_score:.1f} out of 100**."

    clinical_report = f"""
## RADIOLOGICAL AI ANALYSIS REPORT
**DATE**: {timestamp}  
**PRIMARY AI DIAGNOSIS**: **{formatted_label}**  
**MC CONFIDENCE**: {probs[predicted_class]*100:.2f}%  

### 🔬 RADIOMICS OBSERVATIONS (EASY TO UNDERSTAND)
- **Tumor Size**: {obs_size}
- **Tumor Texture**: {obs_texture}
- **Tumor Edges**: {obs_edges}
- **Severity Score**: {obs_sev}

---
{medical_inference[predicted_label]}
"""'''

if "obs_size = " not in code:
    code = code.replace(old_report, new_report)

# Now find the return statement in predict() and wrap all figures in fig_to_image()
old_return = '''    return (
        label_dict, cam_image, fig_clahe, fig_radar, fig_hist, 
        fig_feat, fig_saliency, fig_3d, fig_gauge, fig_guided, 
        fig_var, fig_contour, fig_channels, fig_sev, fig_edges, 
        fig_profile, fig_water, fig_saliency_ratio, clinical_report, status_html
    )'''

new_return = '''    return (
        label_dict, cam_image, fig_to_image(fig_clahe), fig_to_image(fig_radar), fig_to_image(fig_hist), 
        fig_to_image(fig_feat), fig_to_image(fig_saliency), fig_to_image(fig_3d), fig_to_image(fig_gauge), fig_to_image(fig_guided), 
        fig_to_image(fig_var), fig_to_image(fig_contour), fig_to_image(fig_channels), fig_to_image(fig_sev), fig_to_image(fig_edges), 
        fig_to_image(fig_profile), fig_to_image(fig_water), fig_to_image(fig_saliency_ratio), clinical_report, status_html
    )'''

if "fig_to_image(fig_clahe)" not in code:
    code = code.replace(old_return, new_return)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("app.py successfully updated.")
