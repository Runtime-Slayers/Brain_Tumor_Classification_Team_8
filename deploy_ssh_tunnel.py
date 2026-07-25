import subprocess
import re
import sys

print("Initializing secure public deployment tunnel via Pinggy over standard HTTPS Port 443...")
print("(This port is universally allowed by all firewalls and networks)")

cmd = [
    "ssh",
    "-p", "443",
    "-R0:127.0.0.1:7860",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ServerAliveInterval=30",
    "a.pinggy.io"
]

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace',
    bufsize=1
)

url_pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.a\.pinggy\.link")

for line in process.stdout:
    line_clean = line.strip()
    match = url_pattern.search(line_clean)
    if match:
        url = match.group(0)
        print("\n" + "="*65)
        print(f"FIREWALL-PROOF PUBLIC DEPLOYMENT URL READY:")
        print(url)
        print("="*65 + "\n")
        sys.stdout.flush()
    else:
        # Print normal ssh greetings / info
        if any(keyword in line_clean.lower() for keyword in ["http", "tcp", "tunnel", "forward", "port", "pinggy", "url"]):
            print(f"[TUNNEL]: {line_clean}")
            sys.stdout.flush()

process.wait()
