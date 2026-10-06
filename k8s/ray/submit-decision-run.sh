#!/usr/bin/env bash
# Submits a decision-model run to the dd-decision RayCluster from your laptop:
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
echo "==> staging $(du -sh "$STAGE/app" | cut -f1) into $HEAD:/tmp/dd-app"
oc exec -n "$NS" "$HEAD" -c ray-head -- rm -rf /tmp/dd-app
oc rsync -q -n "$NS" -c ray-head "$STAGE/app/" "$HEAD:/tmp/dd-app" ; rm -rf "$STAGE"

RUNTIME_ENV="$(cat <<JSON
{"env_vars": {
   "DGEMMA_HEADLESS_SERVICE": "dgemma-headless.dgemma.svc.cluster.local", "DGEMMA_PORT": "8011",
   "DGEMMA_MODEL": "dgemma", "DGEMMA_TIMEOUT_S": "${DGEMMA_TIMEOUT_S:-30}", "DGEMMA_IMAGE_SCALE": "${DGEMMA_IMAGE_SCALE:-4}",
   "DGEMMA_INCLUDE_NATIVE_FRAME": "${DGEMMA_INCLUDE_NATIVE_FRAME:-0}", "DGEMMA_THINK_WHEN_STUCK": "${DGEMMA_THINK_WHEN_STUCK:-0}",
   "DD_AUTO_FLIGHT": "${DD_AUTO_FLIGHT:-align}", "DD_BOSS_TEXT": "${DD_BOSS_TEXT:-hybrid}", "DD_AUTO_BOSS": "${DD_AUTO_BOSS:-0}", "DD_SAVE_AREAS": "${DD_SAVE_AREAS:-}", "DD_AUTO_MIN_SEEN": "${DD_AUTO_MIN_SEEN:-6}", "DD_AUTO_ELBOW": "${DD_AUTO_ELBOW:-0}", "DD_PLAN_FIGHTS": "${DD_PLAN_FIGHTS:-}", "DD_PLAN_ANY_ENEMY": "${DD_PLAN_ANY_ENEMY:-}", "DD_PLAN_PROGRESS_W": "${DD_PLAN_PROGRESS_W:-1.0}", "DD_PLAN_ANY_ENEMY_PX": "${DD_PLAN_ANY_ENEMY_PX:-90}", "DD_PLAN_ESCAPE": "${DD_PLAN_ESCAPE-86}", "DD_NAV_PLAN": "${DD_NAV_PLAN:-1}", "DD_NAV_STALL_DECISIONS": "${DD_NAV_STALL_DECISIONS:-30}", "DD_NAV_WINDOW_STEPS": "${DD_NAV_WINDOW_STEPS:-1500}", "DD_NAV_STALL_STEPS": "${DD_NAV_STALL_STEPS:-900}", "DD_NAV_TIME_S": "${DD_NAV_TIME_S:-120}", "DD_NAV_PARTIAL_PX": "${DD_NAV_PARTIAL_PX:-24}", "DD_NAV_OPEN_PX": "${DD_NAV_OPEN_PX:-96}", "DD_PLAN_HP_W": "${DD_PLAN_HP_W:-12}", "DD_PLAN_MAX": "${DD_PLAN_MAX:-80}", "DD_PLAN_BEAM_M4": "${DD_PLAN_BEAM_M4:-40}", "DD_PLAN_ESCAPE_PX": "${DD_PLAN_ESCAPE_PX:-150}", "DD_PLAN_ESCAPE_DEPTH": "${DD_PLAN_ESCAPE_DEPTH:-420}", "DD_PLAN_ANY_ENEMY_ALWAYS": "${DD_PLAN_ANY_ENEMY_ALWAYS:-}", "DD_PLAN_SAFE_EXIT_PX": "${DD_PLAN_SAFE_EXIT_PX:-28}", "DD_PLAN_PARTIAL_STEPS": "${DD_PLAN_PARTIAL_STEPS:-60}", "DD_PLAN_BEAM": "${DD_PLAN_BEAM:-24}", "DD_PLAN_DEPTH": "${DD_PLAN_DEPTH:-260}", "DD_PLAN_TIME_S": "${DD_PLAN_TIME_S:-600}",
   "S3_BUCKET_NAME": "sessions", "S3_ENDPOINT_URL": "http://minio.minio.svc:9000",
   "S3_ACCESS_KEY_ID": "minio", "S3_SECRET_ACCESS_KEY": "minio123", "S3_REGION_NAME": "us-east-1"},
 "pip": ["boto3"],
 "excludes": ["__pycache__/", "*.pyc"]}
JSON
)"
echo "==> submitting"
oc exec -n "$NS" "$HEAD" -c ray-head -- bash -lc "cd /tmp/dd-app && ray job submit --address http://127.0.0.1:8265 --no-wait \
  --submission-id dd-$RUN_TAG --working-dir /tmp/dd-app --runtime-env-json '$RUNTIME_ENV' \
  -- python run-decision-dd.py --run-tag $RUN_TAG $*" 2>&1 | grep -vE "^\s*$"
echo
echo "follow:  oc exec -n $NS $HEAD -c ray-head -- ray job logs --address http://127.0.0.1:8265 -f dd-$RUN_TAG"
echo "status:  oc exec -n $NS $HEAD -c ray-head -- ray job status --address http://127.0.0.1:8265 dd-$RUN_TAG"
