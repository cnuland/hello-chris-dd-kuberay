"""Stand-in for the DiffusionGemma structured server: answers /v1/systemone with a random-ish choice.
Usage: python tests/fake_decision_server.py 18011   (then point the runner at http://127.0.0.1:18011)"""
import json, sys, random, os
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1]) if len(sys.argv) > 1 else 18011


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n))
        names = list(body["questions"]["action"]["criteria"].keys())
        pick = os.environ.get("FAKE_ACTION") or random.choice(["right", "right", "a", "b", "jump_kick"])
        probs = {nm: 0.02 for nm in names}; probs[pick] = 0.8
        out = {"answers": {"action": {"type": "choice", "choice": pick, "confidence": 0.8, "probabilities": probs}}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)


HTTPServer(("127.0.0.1", port), H).serve_forever()
