#!/usr/bin/env bash
# Deploys the 4-replica DiffusionGemma decision-model service to the logged-in cluster
# and smoke-tests one round trip. Re-runnable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=dgemma

oc apply -f "$HERE/namespace.yaml"
# The upstream vLLM image expects to run as its own UID (root) and writes under $HOME.
# Prototype-only: let this one service account bypass the restricted SCC.
oc adm policy add-scc-to-user anyuid -z dgemma -n "$NS" >/dev/null
oc apply -f "$HERE/service.yaml" -f "$HERE/deployment.yaml"

echo "==> Waiting for GPU nodes to be schedulable"
oc get nodes -l workload=gpu -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu' || true

echo "==> Waiting for rollout (first start downloads ~27 GB per pod; 10-20 min)"
oc rollout status deployment/dgemma -n "$NS" --timeout=40m
oc get pods -n "$NS" -o wide

echo "==> Smoke test through the ClusterIP service"
oc run dgemma-smoke -n "$NS" --rm -i --restart=Never --image=registry.access.redhat.com/ubi9/ubi-minimal -- \
  curl -sS http://dgemma:8011/v1/systemone -H 'content-type: application/json' -d '{
    "model": "dgemma",
    "state": {"game": "Double Dragon", "lives": 3, "score": 0, "enemy": "one enemy approaching from the right"},
    "questions": {"action": {"type": "choice", "instructions": "Best next input?",
      "options": [{"name":"left","description":"move left"},{"name":"right","description":"move right"},
                  {"name":"a","description":"attack"}]}}}' ; echo
