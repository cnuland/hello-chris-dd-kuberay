"""Latency benchmark against the decision service with a realistic observation payload.
Usage (inside the cluster): python tests/bench_latency.py http://dgemma.dgemma.svc:8011 [calls]"""
import json, sys, time, os
import numpy as np, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from decision_client import DecisionModelClient

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8011"
calls = int(sys.argv[2]) if len(sys.argv) > 2 else 4
c = DecisionModelClient(endpoints=[url], timeout_s=180)
obs = (np.random.rand(128, 40, 3) * 255).astype(np.uint8)
info = {"score": 120, "lives": 3, "level": 15, "steps": 7, "levels_completed": 0}
body = c.build_request(obs, info)
print("payload_bytes", len(json.dumps(body)))
for label, extra in (("default", {}), ("samples=1", {"samples": 1}), ("samples=1,no-image", {"samples": 1, "images": []})):
    b = dict(body, **extra)
    ts = []
    for i in range(calls):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/v1/systemone", json=b, timeout=180)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        a = r.json()["answers"]["action"]
        ts.append(dt)
        print(f"{label:20s} call{i}: {dt:6.2f}s choice={a['choice']:<10s} conf={a['confidence']:.2f} usage={r.json().get('usage')}")
    print(f"{label:20s} -> mean of calls 1..n: {sum(ts[1:]) / max(len(ts) - 1, 1):.2f}s")
