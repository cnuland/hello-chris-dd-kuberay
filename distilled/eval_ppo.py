"""Evaluate a PPO checkpoint (old RLlib API stack, policies/default_policy) on DDEnv with the decision runs' metrics.

Episode records go to s3://sessions/decision_runs/<run_tag>/episodes/w###_ep##.json so analyze_runs.py and the ad-hoc
notebooks can compare the PPO policy with the decision-model runs directly.
Usage (inside the Ray job):  python eval_ppo.py --checkpoint s3://sessions/ray_tune_results/.../checkpoint_000002 \
                                --num-envs 10 --episodes-per-env 1 --max-steps 6000 --run-tag ppo-eval-...
"""
import argparse, json, os, sys, time, importlib.util
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402  app/ddenv.py + app/custom_cnn_model.py
from ddenv import DDEnv
from radar import ScrollTracker
_rd = importlib.util.spec_from_file_location("run_decision_dd", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run-decision-dd.py"))
_mod = importlib.util.module_from_spec(_rd); _rd.loader.exec_module(_mod)
build_env_config, LEVEL_NAMES = _mod.build_env_config, _mod.LEVEL_NAMES


def s3_client():
    import boto3
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"), aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"), region_name=os.environ.get("S3_REGION_NAME", "us-east-1"))


def fetch_checkpoint(uri: str, dest: str) -> str:
    """s3://bucket/prefix -> local dir (whole tree). A local path is returned as is."""
    if not uri.startswith("s3://"):
        return uri
    bucket, prefix = uri[5:].split("/", 1); prefix = prefix.rstrip("/") + "/"
    s3 = s3_client(); os.makedirs(dest, exist_ok=True)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            path = os.path.join(dest, rel); os.makedirs(os.path.dirname(path), exist_ok=True)
            s3.download_file(bucket, o["Key"], path)
    return dest


def load_policy(ckpt_dir: str):
    from ray.rllib.policy.policy import Policy
    from ray.rllib.models import ModelCatalog
    from custom_cnn_model import CustomCNNModel
    ModelCatalog.register_custom_model("custom_cnn", CustomCNNModel)   # the checkpoint's policy config names it
    pol_dir = os.path.join(ckpt_dir, "policies", "default_policy")
    return Policy.from_checkpoint(pol_dir if os.path.isdir(pol_dir) else ckpt_dir)


def run_episodes(idx: int, env_config: Dict[str, Any], ckpt_uri: str, episodes: int, run_tag: str, explore: bool) -> List[Dict[str, Any]]:
    ckpt_dir = fetch_checkpoint(ckpt_uri, f"/tmp/ppo_ckpt_{idx}")   # workers run on other nodes: fetch there, not on the driver
    policy = load_policy(ckpt_dir)
    env = DDEnv(env_config)
    s3 = s3_client() if os.environ.get("S3_ENDPOINT_URL") else None
    out = []
    for ep in range(episodes):
        obs, info = env.reset(); scroll = ScrollTracker(env.pyboy)
        start_level = int(info.get("level", env.pyboy.get_memory_value(0xE110)))
        levels = [start_level]; steps = 0; ep_return = 0.0; t0 = time.time(); best_score = 0
        while True:
            action, _, _ = policy.compute_single_action(obs, explore=explore)
            obs, reward, term, trunc, info = env.step(int(action)); scroll.update(); steps += 1; ep_return += float(reward)
            lvl = int(info.get("level", 0))
            if lvl != levels[-1]:
                levels.append(lvl)
            best_score = max(best_score, int(info.get("score", 0)))
            if term or trunc:
                break
        rec = {"worker": idx, "episode": ep, "run_tag": run_tag, "policy": "ppo", "checkpoint": os.environ.get("PPO_CHECKPOINT", ckpt_dir),
               "start_level": start_level, "start_level_name": LEVEL_NAMES.get(start_level, f"unknown-{start_level}"),
               "distance_px": int(scroll.distance_px), "vertical_px": int(scroll.vertical_px), "final_score": int(info.get("score", 0)),
               "max_score": best_score, "final_lives": int(info.get("lives", 0)), "steps": steps, "terminated": bool(term), "truncated": bool(trunc),
               "return": ep_return, "levels_visited": levels, "levels_visited_names": [LEVEL_NAMES.get(l, f"unknown-{l}") for l in levels],
               "wall_s": time.time() - t0, "reflex_steps": 0, "fallback_steps": 0, "model_ok_steps": steps}
        out.append(rec)
        if s3:
            s3.put_object(Bucket=os.environ.get("S3_BUCKET_NAME", "sessions"), Key=f"decision_runs/{run_tag}/episodes/w{idx:03d}_ep{ep:02d}.json",
                          Body=json.dumps(rec).encode())
        print(f"worker {idx} ep {ep}: start {rec['start_level_name']} dist {rec['distance_px']} score {rec['final_score']} lives {rec['final_lives']} steps {steps}", flush=True)
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--num-envs", type=int, default=10); p.add_argument("--episodes-per-env", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=6000); p.add_argument("--run-tag", default=f"ppo-eval-{time.strftime('%m%d-%H%M')}")
    p.add_argument("--init-states", default="dd.gb.state,lvl2-dd.gb.state,m3-start-model.state"); p.add_argument("--explore", action="store_true")
    p.add_argument("--local", action="store_true", help="run in-process without Ray")
    args = p.parse_args()
    os.environ["PPO_CHECKPOINT"] = args.checkpoint
    ckpt = args.checkpoint
    env_config = build_env_config(args.max_steps, save_video=False, run_tag=args.run_tag, init_states=[s.strip() for s in args.init_states.split(",") if s.strip()])
    if args.local:
        results = run_episodes(0, env_config, ckpt, args.episodes_per_env, args.run_tag, args.explore)
    else:
        import ray
        ray.init(address="auto")
        remote = ray.remote(num_cpus=1)(run_episodes)
        results = sum(ray.get([remote.remote(i, env_config, ckpt, args.episodes_per_env, args.run_tag, args.explore) for i in range(args.num_envs)]), [])
    d = [r["distance_px"] for r in results]
    print(json.dumps({"run summary": args.run_tag, "episodes": len(results), "distance_px_mean": float(np.mean(d)), "distance_px_max": int(max(d)),
                      "score_mean": float(np.mean([r["final_score"] for r in results])), "died_fraction": float(np.mean([r["final_lives"] == 0 for r in results]))}, indent=1))


if __name__ == "__main__":
    main()
