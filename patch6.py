import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the pill shape ([...]) with the cloud shape )...(
# But only for the cloud nodes.
content = content.replace('([☁️', ')☁️')
content = content.replace(']):::cloudNote', '(`:::cloudNote')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully converted nodes to actual Mermaid clouds!')
