import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Revert the huge font size for Attention Saliency Ratio
content = content.replace(
    "ax_sr.text(val + 2, 0, f\"{val:.1f}%\", va='center', weight='bold', color='white', fontsize=24)",
    "ax_sr.text(val + 2, 0, f\"{val:.1f}%\", va='center', weight='bold', color='white')"
)

content = content.replace(
    "ax_sr.set_title(\"Attention Saliency Ratio\", weight='bold', fontsize=16)",
    "ax_sr.set_title(\"Attention Saliency Ratio\", weight='bold')"
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully reverted font size!')
