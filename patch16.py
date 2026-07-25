import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = 'val = saliency_ratio * 100'
insertion = """
    # Create an image with ONLY the percentage printed for Saliency Ratio
    fig_saliency_ratio, ax_sr = plt.subplots(figsize=(5, 3.5))
    fig_saliency_ratio.patch.set_facecolor('#0f172a')
    ax_sr.set_facecolor('#0f172a')
    ax_sr.set_xlim(0, 1)
    ax_sr.set_ylim(0, 1)
    ax_sr.text(0.5, 0.5, f"{val:.1f}%", va='center', ha='center', weight='bold', color='#10b981', fontsize=48)
    ax_sr.axis('off')
"""

content = content.replace(target, target + '\n' + insertion)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully injected fig_saliency_ratio code!')
