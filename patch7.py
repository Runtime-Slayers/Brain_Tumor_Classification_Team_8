import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I need to wrap the text inside the cloud nodes with double quotes!
# Currently it looks like: A_cloud)☁️ Dim: 512x512x3 • Feat: RGB Pixels<br/>Importance: Preserves original unaltered anatomy(:::cloudNote
# I want: A_cloud)"☁️ Dim: 512x512x3 • Feat: RGB Pixels<br/>Importance: Preserves original unaltered anatomy"(:::cloudNote

def quote_cloud(match):
    # match.group(1) is the node name (e.g. A_cloud)
    # match.group(2) is the text inside
    return f'{match.group(1)})"{match.group(2)}"('

# Regex to find: NodeName)Text(
# and replace with: NodeName)"Text"(
content = re.sub(r'([A-Z]_cloud)\)(.*?)\(', quote_cloud, content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully wrapped cloud text in quotes!')
