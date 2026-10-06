#!/usr/bin/env bash
# PPO fine-tune (run-ray-dd.py) warm-started from a distilled policy: BC_INIT_WEIGHTS=s3://sessions/models/bc_policy_<run_tag>.pt
# Usage: k8s/ray/submit-ppo-warmstart.sh <decision_run_tag>
set -euo pipefail
NS="${NS:-hello-chris-ai-4}"; RC="${RC:-dd-decision}"
RUN_TAG="${1:?decision run_tag whose bc_policy to start from}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HEAD="$(oc get pods -n "$NS" -l "ray.io/cluster=$RC,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}')"
STAGE="$(mktemp -d)"; mkdir -p "$STAGE/app/ignored"; cp "$HERE"/app/*.py "$STAGE/app/"; cp "$HERE"/app/ignored/*.gb "$HERE"/app/ignored/*.state "$STAGE/app/ignored/"
oc exec -n "$NS" "$HEAD" -c ray-head -- rm -rf /tmp/dd-ppo; oc rsync -q -n "$NS" -c ray-head "$STAGE/app/" "$HEAD:/tmp/dd-ppo"; rm -rf "$STAGE"
ENVJSON="$(cat <<JSON
{"env_vars": {"S3_BUCKET_NAME": "sessions", "S3_ENDPOINT_URL": "http://minio.minio.svc:9000", "S3_ACCESS_KEY_ID": "minio",
  "S3_SECRET_ACCESS_KEY": "minio123", "S3_REGION_NAME": "us-east-1",
  "BC_INIT_WEIGHTS": "s3://sessions/models/bc_policy_${RUN_TAG}.pt"},
 "pip": ["boto3"], "excludes": ["__pycache__/"]}
JSON
)"
oc exec -n "$NS" "$HEAD" -c ray-head -- bash -lc "cd /tmp/dd-ppo && ray job submit --address http://127.0.0.1:8265 --no-wait \
  --submission-id ppo-$RUN_TAG-$(date +%H%M) --working-dir /tmp/dd-ppo --runtime-env-json '$ENVJSON' -- python run-ray-dd.py" 2>&1 | grep -E "submitted|Error|error"
echo "follow: oc exec -n $NS $HEAD -c ray-head -- ray job list --address http://127.0.0.1:8265 | grep ppo-"
