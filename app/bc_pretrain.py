# bc_pretrain.py
"""
Behaviour-cloning pretraining: distil the DiffusionGemma decision model into the PPO policy net.

Input : the .npz traces run-decision-dd.py uploads (obs (T,128,40,3) uint8 + teacher_probs (T,9)).
Output: a state_dict for app/custom_cnn_model.py's CustomCNNModel (conv_layers.* and fc.*), which
        run-ray-dd.py can load as a warm start via BC_INIT_WEIGHTS=<s3://... or local path>.

Loss  : KL(teacher || student) on the full 9-way probability vector (soft targets), which is what the
        decision model's calibrated logprobs are good for. --hard switches to cross-entropy on argmax.

Usage:
    python bc_pretrain.py --s3-prefix decision_runs/<run_tag>/traces/ --epochs 5 --out bc_policy.pt
    python bc_pretrain.py --local-glob "traces/*.npz" --out bc_policy.pt
Env: S3_BUCKET_NAME, S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY (same as the runner).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import os
import sys
import time
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bc_pretrain")


class StudentPolicy(nn.Module):
    """Exactly CustomCNNModel's conv trunk + policy head, so the state_dict keys line up."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv_layers(torch.zeros(1, 3, 128, 40)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, num_actions))
        self.value_branch = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,H,W,C) uint8
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.fc(self.conv_layers(x))


def iter_traces_s3(prefix: str) -> Iterable[Tuple[str, bytes]]:
    import boto3
    params = {k: v for k, v in {
        "aws_access_key_id": os.environ.get("S3_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("S3_SECRET_ACCESS_KEY"),
        "region_name": os.environ.get("S3_REGION_NAME"),
        "endpoint_url": os.environ.get("S3_ENDPOINT_URL"),
    }.items() if v}
    s3 = boto3.client("s3", **params)
    bucket = os.environ["S3_BUCKET_NAME"]
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".npz"):
                yield o["Key"], s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()


def iter_traces_local(pattern: str) -> Iterable[Tuple[str, bytes]]:
    for p in sorted(glob.glob(pattern)):
        yield p, open(p, "rb").read()


def load_dataset(traces: Iterable[Tuple[str, bytes]], min_confidence: float, only_ok: bool):
    obs_parts: List[np.ndarray] = []
    prob_parts: List[np.ndarray] = []
    names = None
    n_files = 0
    for key, blob in traces:
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        if "obs" not in z:
            log.warning("%s has no observations (recorded with --no-obs-in-traces); skipping", key)
            continue
        steps, probs, obs = z["steps"], z["teacher_probs"], z["obs"]
        if names is None:
            names = [str(n) for n in z["action_names"]]
        keep = np.ones(len(steps), dtype=bool)
        if only_ok:
            keep &= steps[:, 4] > 0.5          # column 4 = model answered (not a random fallback)
        if min_confidence > 0:
            keep &= steps[:, 2] >= min_confidence
        obs_parts.append(obs[keep]); prob_parts.append(probs[keep]); n_files += 1
    if not obs_parts:
        raise SystemExit("no usable traces found")
    X = np.concatenate(obs_parts); P = np.concatenate(prob_parts).astype(np.float32)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-8, None)
    log.info("dataset: %d frames from %d traces, actions=%s", len(X), n_files, names)
    return X, P, names


def train(X: np.ndarray, P: np.ndarray, num_actions: int, epochs: int, batch: int, lr: float, hard: bool, device: str):
    model = StudentPolicy(num_actions).to(device)
    opt = torch.optim.Adam(list(model.conv_layers.parameters()) + list(model.fc.parameters()), lr=lr)
    n = len(X); idx = np.arange(n)
    n_val = max(1, int(0.05 * n)); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X); Pt = torch.from_numpy(P)
    for ep in range(epochs):
        np.random.shuffle(tr_idx); model.train(); tot = 0.0; t0 = time.time()
        for i in range(0, len(tr_idx), batch):
            b = torch.from_numpy(tr_idx[i:i + batch])
            logits = model(Xt[b].to(device)); target = Pt[b].to(device)
            if hard:
                loss = F.cross_entropy(logits, target.argmax(1))
            else:
                loss = F.kl_div(F.log_softmax(logits, 1), target, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = model(Xt[val_idx].to(device)); vt = Pt[val_idx].to(device)
            val_kl = F.kl_div(F.log_softmax(vl, 1), vt, reduction="batchmean").item()
            agree = (vl.argmax(1) == vt.argmax(1)).float().mean().item()
        log.info("epoch %d/%d train_loss=%.4f val_kl=%.4f val_argmax_agreement=%.3f (%.0fs)",
                 ep + 1, epochs, tot / max(len(tr_idx), 1), val_kl, agree, time.time() - t0)
    return model, {"val_kl": val_kl, "val_argmax_agreement": agree, "frames": int(n)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--s3-prefix", help="e.g. decision_runs/<run_tag>/traces/")
    src.add_argument("--local-glob", help="e.g. 'traces/*.npz'")
    ap.add_argument("--out", default="bc_policy.pt")
    ap.add_argument("--upload-key", default=None, help="also upload to s3://$S3_BUCKET_NAME/<key>")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-confidence", type=float, default=0.0, help="drop frames where the teacher was unsure")
    ap.add_argument("--include-fallbacks", action="store_true", help="keep random-fallback steps (default: drop)")
    ap.add_argument("--hard", action="store_true", help="cross-entropy on argmax instead of KL to soft targets")
    args = ap.parse_args(argv)

    if args.s3_prefix:
        # several runs may be combined: --s3-prefix decision_runs/a/traces/,decision_runs/b/traces/
        prefixes = [p.strip() for p in args.s3_prefix.split(",") if p.strip()]
        traces = (t for p in prefixes for t in iter_traces_s3(p))
    else:
        traces = iter_traces_local(args.local_glob)
    X, P, names = load_dataset(traces, args.min_confidence, only_ok=not args.include_fallbacks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metrics = train(X, P, len(names), args.epochs, args.batch, args.lr, args.hard, device)

    sd = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith(("conv_layers.", "fc."))}
    payload = {"state_dict": sd, "action_names": names, "metrics": metrics,
               "source": args.s3_prefix or args.local_glob, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    torch.save(payload, args.out)
    log.info("saved %s (%d tensors) metrics=%s", args.out, len(sd), json.dumps(metrics))
    if args.upload_key:
        import boto3
        params = {k: v for k, v in {
            "aws_access_key_id": os.environ.get("S3_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("S3_SECRET_ACCESS_KEY"),
            "region_name": os.environ.get("S3_REGION_NAME"),
            "endpoint_url": os.environ.get("S3_ENDPOINT_URL"),
        }.items() if v}
        boto3.client("s3", **params).upload_file(args.out, os.environ["S3_BUCKET_NAME"], args.upload_key)
        log.info("uploaded to s3://%s/%s", os.environ["S3_BUCKET_NAME"], args.upload_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
