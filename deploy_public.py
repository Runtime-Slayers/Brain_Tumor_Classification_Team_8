from pycloudflared import try_cloudflare
import time
import sys
import codecs

# Force utf-8 printing just in case
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("Initiating Cloudflare Quick Tunnel for port 7860...")
tunnel = try_cloudflare(port=7860)
url = tunnel.tunnel
print("\n" + "="*60)
print(f"🌐 PUBLIC DEPLOYMENT URL READY: {url}")
print("="*60 + "\n")
print("Tunnel active. Do not close this process...")

while True:
    time.sleep(60)
