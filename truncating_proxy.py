"""HTTP proxy to the public S3 bucket that truncates a fraction of chunk responses.

Reproduces the failure mode in villa #1666: the object is fine, but the
response body stops short of Content-Length (aiohttp then raises
ClientPayloadError / ContentLengthError). Usage: python truncating_proxy.py PORT FRACTION
"""
import http.server, socketserver, sys, random, urllib.request, urllib.error, threading
PORT=int(sys.argv[1]); FRACTION=float(sys.argv[2]); SEED=int(sys.argv[3]) if len(sys.argv)>3 else 0
UPSTREAM="https://vesuvius-challenge-open-data.s3.amazonaws.com"
rng=random.Random(SEED); lock=threading.Lock(); stats={"ok":0,"truncated":0,"errors":0}
class H(http.server.BaseHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    def log_message(self,*a): pass
    def do_HEAD(self): self._do(head=True)
    def do_GET(self): self._do(head=False)
    def _do(self, head):
        url=UPSTREAM+self.path
        try:
            req=urllib.request.Request(url, method="HEAD" if head else "GET")
            with urllib.request.urlopen(req, timeout=60) as r:
                body=b"" if head else r.read(); status=r.status; ctype=r.headers.get("Content-Type","application/octet-stream")
        except urllib.error.HTTPError as e:
            body=e.read(); status=e.code; ctype="application/xml"
        truncate = (not head) and status==200 and len(body)>4096 and rng.random()<FRACTION
        self.send_response(status); self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(body))); self.end_headers()
        if head: return
        with lock:
            if truncate:
                stats["truncated"]+=1; cut=len(body)//3
                self.wfile.write(body[:cut]); self.wfile.flush()
                self.close_connection=True
                self.connection.close()   # body stops short of Content-Length
                sys.stderr.write(f"[proxy] TRUNCATED {self.path} ({cut}/{len(body)} bytes)\n"); sys.stderr.flush()
                return
            stats["ok"]+=1
        self.wfile.write(body)
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads=True
sys.stderr.write(f"[proxy] listening on {PORT}, truncating {FRACTION:.0%} of chunk responses (seed {SEED})\n"); sys.stderr.flush()
S(("127.0.0.1",PORT),H).serve_forever()
