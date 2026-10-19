#!/usr/bin/env bash
# Distil a decision run's traces into the PPO policy net, on the Ray cluster.
# Usage: distilled/k8s/ray/submit-bc-pretrain.sh <run_tag> [extra bc_pretrain.py args]
#   -> s3://sessions/models/bc_policy_<run_tag>.pt, loadable via BC_INIT_WEIGHTS in run-ray-dd.py
set -euo pipefail
NS="${NS:-hello-chris-ai-4}"; RC="${RC:-dd-decision}"
RUN_TAG="${1:?run_tag}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root (script lives in distilled/k8s/ray)
# MinIO keys come from the cluster Secret (never hard-coded): override with S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY
S3_KEY="${S3_ACCESS_KEY_ID:-$(oc get secret minio-root-user -n minio -o jsonpath='{.data.MINIO_ROOT_USER}' | base64 -d)}"
S3_SECRET="${S3_SECRET_ACCESS_KEY:-$(oc get secret minio-root-user -n minio -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 -d)}"
[ -n "$S3_SECRET" ] || { echo "no MinIO secret (set S3_SECRET_ACCESS_KEY or create minio/minio-root-user)"; exit 1; }

HEAD="$(oc get pods -n "$NS" -l "ray.io/cluster=$RC,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}')"
STAGE="$(mktemp -d)"; mkdir -p "$STAGE/app"; cp "$HERE"/distilled/*.py "$HERE"/app/ddenv.py "$HERE"/app/custom_cnn_model.py "$STAGE/app/"
oc exec -n "$NS" "$HEAD" -c ray-head -- rm -rf /tmp/dd-bc; oc rsync -q -n "$NS" -c ray-head "$STAGE/app/" "$HEAD:/tmp/dd-bc"; rm -rf "$STAGE"
ENVJSON='{"env_vars": {"S3_BUCKET_NAME": "sessions", "S3_ENDPOINT_URL": "http://minio.minio.svc:9000", "S3_ACCESS_KEY_ID": "__S3_KEY__", "S3_SECRET_ACCESS_KEY": "__S3_SECRET__", "S3_REGION_NAME": "us-east-1"}, "pip": ["boto3"], "excludes": ["__pycache__/"]}'
ENVJSON="${ENVJSON//__S3_KEY__/$S3_KEY}"; ENVJSON="${ENVJSON//__S3_SECRET__/$S3_SECRET}"
oc exec -n "$NS" "$HEAD" -c ray-head -- bash -lc "cd /tmp/dd-bc && ray job submit --address http://127.0.0.1:8265 --no-wait \
  --submission-id bc-$RUN_TAG --entrypoint-num-cpus 8 --entrypoint-memory $((24*1024*1024*1024)) \
  --working-dir /tmp/dd-bc --runtime-env-json '$ENVJSON' \
  -- python bc_pretrain.py --s3-prefix decision_runs/$RUN_TAG/traces/ --epochs 5 --batch 256 \
     --out bc_policy.pt --upload-key models/bc_policy_$RUN_TAG.pt $*" 2>&1 | grep -E "submitted|Error|error"
echo "follow: oc exec -n $NS $HEAD -c ray-head -- ray job logs --address http://127.0.0.1:8265 -f bc-$RUN_TAG"
