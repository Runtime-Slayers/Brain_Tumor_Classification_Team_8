import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

broken_code = """      # Create an image with ONLY the percentage printed for Saliency Ratio
      fig_saliency_ratio, ax_sr = plt.subplots(figsize=(5, 3.5))
      fig_saliency_ratio.patch.set_facecolor('#0f172a')
      ax_sr.set_facecolor('#0f172a')
      ax_sr.text(0.5, 0.5, f"{val:.1f}%", va='center', ha='center', weight='bold', color='#10b981', fontsize=48)
      ax_sr.axis('off')
      plt.tight_layout()"""

fixed_code = """      # Create an image with ONLY the percentage printed for Saliency Ratio
      fig_saliency_ratio, ax_sr = plt.subplots(figsize=(5, 3.5))
      fig_saliency_ratio.patch.set_facecolor('#0f172a')
      ax_sr.set_facecolor('#0f172a')
      ax_sr.set_xlim(0, 1)
      ax_sr.set_ylim(0, 1)
      ax_sr.text(0.5, 0.5, f"{val:.1f}%", va='center', ha='center', weight='bold', color='#10b981', fontsize=48)
      ax_sr.axis('off')
      # Removed tight_layout() to prevent canvas from collapsing to 0 width"""

content = content.replace(broken_code, fixed_code)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed tight_layout collapsing bug!')
