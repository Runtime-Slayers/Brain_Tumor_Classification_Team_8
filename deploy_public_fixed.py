import os
import sys
import time
import urllib.request
import subprocess
import re
from pathlib import Path
import site

print("Verifying local Gradio server at http://127.0.0.1:7860...")
try:
    with urllib.request.urlopen("http://127.0.0.1:7860", timeout=5) as response:
        print(f"[SUCCESS] Local server responding: HTTP {response.status}")
except Exception as e:
    print(f"[WARNING] Warning during local ping: {e}")

# Find cloudflared executable installed by pycloudflared
cloudflared_path = None
for path in site.getsitepackages() + [site.getusersitepackages()]:
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if "cloudflared" in file and file.endswith(".exe"):
                    cloudflared_path = os.path.join(root, file)
                    break
    if cloudflared_path:
        break

if not cloudflared_path:
    import pycloudflared
    pkg_dir = os.path.dirname(pycloudflared.__file__)
    for file in os.listdir(pkg_dir):
        if "cloudflared" in file and file.endswith(".exe"):
            cloudflared_path = os.path.join(pkg_dir, file)
            break

print(f"Located cloudflared binary at: {cloudflared_path}")

# Force HTTP/2 protocol to bypass network/firewall blocking of QUIC port 7844
cmd = [cloudflared_path, "tunnel", "--url", "http://127.0.0.1:7860", "--protocol", "http2"]
print(f"Launching command: {' '.join(cmd)}")

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace',
    bufsize=1
)

url_pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")

for line in process.stdout:
    line_clean = line.strip()
    match = url_pattern.search(line_clean)
    if match:
        url = match.group(0)
        print("\n" + "="*65)
        print(f"PUBLIC DEPLOYMENT URL READY: {url}")
        print("="*65 + "\n")
        sys.stdout.flush()
    if "error" in line_clean.lower() or "warning" in line_clean.lower():
        print(f"[TUNNEL]: {line_clean}")
        sys.stdout.flush()

process.wait()
