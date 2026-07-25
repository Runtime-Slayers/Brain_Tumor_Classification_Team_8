import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the incorrect cv2.putText on cam_image
content = content.replace(
    '# Overlay Saliency Ratio directly onto the CAM image\n    cv2.putText(cam_image, f"Attention Saliency Ratio: {val:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)\n    \n',
    ''
)
content = content.replace(
    '# Overlay Saliency Ratio directly onto the CAM image\n      cv2.putText(cam_image, f"Attention Saliency Ratio: {val:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)\n      \n',
    ''
)
content = content.replace(
    'cv2.putText(cam_image, f"Attention Saliency Ratio: {val:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)',
    ''
)


# Increase the font size of the text in the actual Attention Saliency Ratio graph
old_text = "ax_sr.text(val + 2, 0, f\"{val:.1f}%\", va='center', weight='bold', color='white')"
new_text = "ax_sr.text(val + 2, 0, f\"{val:.1f}%\", va='center', weight='bold', color='white', fontsize=24)"
content = content.replace(old_text, new_text)

# Also increase title size for better visibility
old_title = 'ax_sr.set_title("Attention Saliency Ratio", weight=\'bold\')'
new_title = 'ax_sr.set_title("Attention Saliency Ratio", weight=\'bold\', fontsize=16)'
content = content.replace(old_title, new_title)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully fixed Saliency Ratio graph and removed defacement of CAM image!')
