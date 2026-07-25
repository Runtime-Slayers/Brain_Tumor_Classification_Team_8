import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Change output_saliency_ratio back to an Image
content = content.replace(
    'output_saliency_ratio = gr.Textbox(label="Attention Saliency Ratio", interactive=False)',
    'output_saliency_ratio = gr.Image(label="Attention Saliency Ratio", interactive=False)'
)

# 2. Add the matplotlib code back to generate an image that ONLY has the percentage text
graph_code = """      saliency_ratio = np.sum(grayscale_cam * tumor_proxy) / (np.sum(grayscale_cam) + 1e-7)
      val = saliency_ratio * 100
      
      # Create an image with ONLY the percentage printed for Saliency Ratio
      fig_saliency_ratio, ax_sr = plt.subplots(figsize=(5, 3.5))
      fig_saliency_ratio.patch.set_facecolor('#0f172a')
      ax_sr.set_facecolor('#0f172a')
      ax_sr.text(0.5, 0.5, f"{val:.1f}%", va='center', ha='center', weight='bold', color='#10b981', fontsize=48)
      ax_sr.axis('off')
      plt.tight_layout()"""

content = content.replace(
    '      saliency_ratio = np.sum(grayscale_cam * tumor_proxy) / (np.sum(grayscale_cam) + 1e-7)\n      val = saliency_ratio * 100\n      \n',
    graph_code + '\n'
)

# 3. Change the return tuple to return fig_to_image(fig_saliency_ratio) again
content = content.replace(
    'f"{val:.1f}%", clinical_report, \nstatus_html',
    'fig_to_image(fig_saliency_ratio), clinical_report, \nstatus_html'
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully converted Saliency Ratio to an image containing only the percentage text!')
