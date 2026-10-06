#!/usr/bin/env bash
# Submits a PPO checkpoint evaluation to the dd-decision RayCluster from your laptop:
#   1. stages app/ (code + ROM + save states, no venv) into the Ray head pod
#   2. `ray job submit` from inside the head with that dir as the job's working_dir, so every
#      Ray worker gets ddenv.py, decision_client.py and ignored/*.gb|*.state
# Usage: k8s/ray/submit-decision-run.sh [extra run-decision-dd.py args]
#   e.g. k8s/ray/submit-decision-run.sh --num-envs 100 --episodes-per-env 1 --max-steps 2048
set -euo pipefail
NS="${NS:-hello-chris-ai-4}"; RC="${RC:-dd-decision}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

HEAD="$(oc get pods -n "$NS" -l "ray.io/cluster=$RC,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}')"
[ -n "$HEAD" ] || { echo "no head pod for RayCluster $RC in $NS"; exit 1; }
echo "==> head pod: $HEAD  run_tag: $RUN_TAG"

STAGE="$(mktemp -d)"; mkdir -p "$STAGE/app/ignored"
cp "$HERE"/app/*.py "$STAGE/app/"
cp "$HERE"/app/ignored/*.gb "$HERE"/app/ignored/*.state "$STAGE/app/ignored/"
echo "==> staging $(du -sh "$STAGE/app" | cut -f1) into $HEAD:/tmp/dd-ppoeval"
oc exec -n "$NS" "$HEAD" -c ray-head -- rm -rf /tmp/dd-ppoeval
oc rsync -q -n "$NS" -c ray-head "$STAGE/app/" "$HEAD:/tmp/dd-ppoeval" ; rm -rf "$STAGE"

RUNTIME_ENV="$(cat <<JSON
{"env_vars": {
   "DGEMMA_HEADLESS_SERVICE": "dgemma-headless.dgemma.svc.cluster.local", "DGEMMA_PORT": "8011",
   "DGEMMA_MODEL": "dgemma", "DGEMMA_TIMEOUT_S": "${DGEMMA_TIMEOUT_S:-30}", "DGEMMA_IMAGE_SCALE": "${DGEMMA_IMAGE_SCALE:-4}",
   "DGEMMA_INCLUDE_NATIVE_FRAME": "${DGEMMA_INCLUDE_NATIVE_FRAME:-0}",
   "S3_BUCKET_NAME": "sessions", "S3_ENDPOINT_URL": "http://minio.minio.svc:9000",
   "S3_ACCESS_KEY_ID": "minio", "S3_SECRET_ACCESS_KEY": "minio123", "S3_REGION_NAME": "us-east-1"},
 "pip": ["boto3"],
 "excludes": ["__pycache__/", "*.pyc"]}
JSON
)"
echo "==> submitting"
oc exec -n "$NS" "$HEAD" -c ray-head -- bash -lc "cd /tmp/dd-ppoeval && ray job submit --address http://127.0.0.1:8265 --no-wait \
  --submission-id ppoeval-$RUN_TAG --working-dir /tmp/dd-ppoeval --runtime-env-json '$RUNTIME_ENV' \
  -- python eval_ppo.py --run-tag $RUN_TAG $*" 2>&1 | grep -vE "^\s*$"
echo
echo "follow:  oc exec -n $NS $HEAD -c ray-head -- ray job logs --address http://127.0.0.1:8265 -f ppoeval-$RUN_TAG"
echo "status:  oc exec -n $NS $HEAD -c ray-head -- ray job status --address http://127.0.0.1:8265 ppoeval-$RUN_TAG"
