import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the Mermaid syntax error by replacing the pipe `|` character inside nodes, 
# which Mermaid interprets as an edge label, causing a syntax error.
content = content.replace(' | ', ' • ')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully removed pipes from Mermaid diagram to fix syntax error!')
