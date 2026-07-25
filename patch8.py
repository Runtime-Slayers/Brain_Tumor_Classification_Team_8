import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the A_cloud)"Text"(:::cloudNote with A_cloud["Text"]@{ shape: cloud }:::cloudNote
# The current format in app.py is A_cloud)"Text"(:::cloudNote
def apply_new_cloud(match):
    return f'{match.group(1)}["{match.group(2)}"]@{{ shape: cloud }}:::'

content = re.sub(r'([A-Z]_cloud\b.*?)\)"(.*?)"\(\s*:::', apply_new_cloud, content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully applied @{ shape: cloud } syntax!')
