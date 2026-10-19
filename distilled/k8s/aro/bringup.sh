#!/usr/bin/env bash
# One-shot ARO bring-up for the DiffusionGemma decision model on A10s:
# login -> GPU MachineSet -> NFD + NVIDIA GPU operator -> dgemma (TP=2) -> local port-forward 18011.
# Uses KUBECONFIG=~/.config/aro-rl/kubeconfig (never touches the user's default context).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
RG="${RG:-openenv-jxdfq}"; CL="${CL:-aro-cluster-jxdfq}"
export KUBECONFIG="$HOME/.config/aro-rl/kubeconfig"
log() { printf '\n==> %s %s\n' "$(date +%H:%M:%S)" "$*"; }

log "login"
API="$(az aro show -g "$RG" -n "$CL" --query apiserverProfile.url -o tsv)"
PW="$(az aro list-credentials -g "$RG" -n "$CL" --query kubeadminPassword -o tsv)"
oc login "$API" -u kubeadmin -p "$PW" --insecure-skip-tls-verify=true >/dev/null; unset PW
oc whoami; oc get clusterversion -o jsonpath='{.items[0].status.desired.version}{"\n"}'

log "GPU MachineSet (Standard_NV72ads_A10_v5, 2x A10)"
"$HERE/gpu-machineset.sh" "${GPU_VM:-Standard_NV72ads_A10_v5}" 1

log "NFD + NVIDIA GPU operator (waits for nvidia.com/gpu allocatable)"
"$REPO/k8s/rosa/install-gpu-operator.sh"

log "dgemma (1 replica, TP=2)"
oc apply -f "$REPO/k8s/dgemma/namespace.yaml"
oc adm policy add-scc-to-user anyuid -z dgemma -n dgemma >/dev/null
oc apply -f "$REPO/k8s/dgemma/service.yaml" -f "$HERE/dgemma-a10.yaml"
oc rollout status deployment/dgemma -n dgemma --timeout=60m
oc get pods -n dgemma -o wide
log "ready"
