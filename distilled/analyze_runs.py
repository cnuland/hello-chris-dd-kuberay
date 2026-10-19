# analyze_runs.py
"""
Compare decision-model runs from the summaries the runner uploads to S3.

    python analyze_runs.py                 # table of every decision_runs/<run_tag>/summary.json
    python analyze_runs.py --tag <run_tag> # per-episode detail for one run (levels visited, scores)
    python analyze_runs.py --json          # machine-readable

Env: S3_BUCKET_NAME, S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY (same as the runner).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List


def s3():
    import boto3
    params = {k: v for k, v in {
        "aws_access_key_id": os.environ.get("S3_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("S3_SECRET_ACCESS_KEY"),
        "region_name": os.environ.get("S3_REGION_NAME"),
        "endpoint_url": os.environ.get("S3_ENDPOINT_URL"),
    }.items() if v}
    return boto3.client("s3", **params), os.environ["S3_BUCKET_NAME"]


def list_summaries(client, bucket) -> List[Dict]:
    out = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="decision_runs/"):
        for o in page.get("Contents", []):
            if o["Key"].endswith("/summary.json"):
                body = client.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()
                s = json.loads(body); s.setdefault("run_tag", o["Key"].split("/")[1]); s["_modified"] = o["LastModified"].isoformat()
                out.append(s)
    return sorted(out, key=lambda s: s["_modified"])


def fmt(v, nd=2):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def print_table(rows: List[Dict]) -> None:
    cols = [("run_tag", 22), ("episodes", 8), ("total_steps", 11), ("score_mean", 10), ("score_max", 9),
            ("distance_px_mean", 9), ("distance_px_max", 8), ("died_fraction", 6), ("model_fallback_fraction", 8),
            ("latency_mean_s", 8), ("confidence_mean", 6), ("wall_time_s", 8)]
    head = {"distance_px_mean": "dist_mean", "distance_px_max": "dist_max", "died_fraction": "died",
            "model_fallback_fraction": "fallback", "latency_mean_s": "lat_s", "confidence_mean": "conf", "wall_time_s": "wall_s"}
    print(" ".join(head.get(c, c).ljust(w) for c, w in cols))
    for r in rows:
        print(" ".join(fmt(r.get(c)).ljust(w)[:w] for c, w in cols))
    print()
    for r in rows:
        reach = r.get("episodes_reaching_level") or {}
        if reach:
            print(f"{r['run_tag']}: episodes reaching level -> " + ", ".join(f"{k}={v}" for k, v in reach.items()))
        cleared = r.get("episodes_game_cleared")
        if cleared:
            print(f"{r['run_tag']}: *** {cleared} episode(s) flagged game_cleared ***")


def episode_detail(client, bucket, tag: str) -> List[Dict]:
    try:
        body = client.get_object(Bucket=bucket, Key=f"decision_runs/{tag}/episodes.jsonl")["Body"].read().decode()
        eps = [json.loads(l) for l in body.splitlines() if l.strip()]
    except client.exceptions.NoSuchKey:
        # run still going (or stopped early): fall back to the per-episode records the workers upload as they finish
        eps = []
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"decision_runs/{tag}/episodes/"):
            for o in page.get("Contents", []):
                eps.append(json.loads(client.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()))
    eps.sort(key=lambda e: (-e.get("distance_px", 0), -(e.get("max_score") or 0)))
    return eps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="show per-episode detail for this run_tag")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    client, bucket = s3()
    if args.tag:
        eps = episode_detail(client, bucket, args.tag)
        if args.json:
            print(json.dumps(eps, indent=1)); return 0
        print(f"{args.tag}: {len(eps)} episodes, best first")
        for e in eps[: args.top]:
            print(f"  w{e['worker']:03d}/ep{e['episode']} steps={e['steps']:5d} dist={e.get('distance_px', '-')} score={e.get('max_score')} "
                  f"lives={e.get('final_lives')} return={e['return']:.2f} levels={'>'.join(e.get('levels_visited_names', []))} "
                  f"fallback={e['fallback_steps']} conf={fmt(e.get('mean_confidence'))}")
        return 0
    rows = list_summaries(client, bucket)
    if args.json:
        print(json.dumps(rows, indent=1)); return 0
    if not rows:
        print("no decision_runs/*/summary.json found in bucket", bucket); return 1
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
