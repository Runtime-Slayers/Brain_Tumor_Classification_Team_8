with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace black needle with white needle
code = code.replace('color="black", shrinkA=0', 'color="white", shrinkA=0')

# Replace black text with white text in gauge
code = code.replace("weight='bold'); plt.tight_layout()", "weight='bold', color='white'); plt.tight_layout()")
code = code.replace("weight='bold'); \n", "weight='bold', color='white'); \n")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Patched gauge colors!')
