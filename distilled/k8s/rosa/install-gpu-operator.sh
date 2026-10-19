#!/usr/bin/env bash
# Installs Node Feature Discovery + NVIDIA GPU Operator from OperatorHub on the
# currently logged-in OpenShift cluster, then waits until nvidia.com/gpu shows up
# as an allocatable resource on the GPU nodes. Safe to re-run.
set -euo pipefail
log() { printf '\n==> %s\n' "$*"; }

default_channel() { oc get packagemanifest "$1" -n openshift-marketplace -o jsonpath='{.status.defaultChannel}'; }
current_csv()     { oc get packagemanifest "$1" -n openshift-marketplace -o jsonpath="{.status.channels[?(@.name==\"$2\")].currentCSV}"; }

wait_csv() { # namespace csv
  for _ in $(seq 1 60); do
    PHASE="$(oc get csv -n "$1" "$2" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    echo "  $(date +%H:%M:%S) $2: ${PHASE:-pending}"
    [[ "$PHASE" == "Succeeded" ]] && return 0
    sleep 10
  done
  echo "CSV $2 did not reach Succeeded"; return 1
}

apply_alm_example() { # namespace csv kind
  oc get csv -n "$1" "$2" -o jsonpath='{.metadata.annotations.alm-examples}' \
    | jq --arg k "$3" '.[] | select(.kind==$k)' | oc apply -n "$1" -f -
}

install_operator() { # namespace package
  local NS="$1" PKG="$2" CH CSV
  CH="$(default_channel "$PKG")"; CSV="$(current_csv "$PKG" "$CH")"
  log "Installing ${PKG} (channel ${CH}, ${CSV}) into ${NS}"
  oc apply -f - <<YAML
apiVersion: v1
kind: Namespace
metadata:
  name: ${NS}
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: ${NS}-og
  namespace: ${NS}
spec:
  targetNamespaces: ["${NS}"]
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: ${PKG}
  namespace: ${NS}
spec:
  channel: "${CH}"
  name: ${PKG}
  source: $(oc get packagemanifest "$PKG" -n openshift-marketplace -o jsonpath='{.status.catalogSource}')
  sourceNamespace: openshift-marketplace
  installPlanApproval: Automatic
  startingCSV: ${CSV}
YAML
  wait_csv "$NS" "$CSV"
  echo "$CSV"
}

# ---- Node Feature Discovery: labels nodes with PCI vendor 10de (NVIDIA) so the GPU operator knows where to run
NFD_CSV="$(install_operator openshift-nfd nfd | tail -1)"
apply_alm_example openshift-nfd "$NFD_CSV" NodeFeatureDiscovery

# ---- NVIDIA GPU Operator: driver, container toolkit, device plugin, DCGM exporter
GPU_CSV="$(install_operator nvidia-gpu-operator gpu-operator-certified | tail -1)"
apply_alm_example nvidia-gpu-operator "$GPU_CSV" ClusterPolicy

log "Waiting for GPU driver to come up and nvidia.com/gpu to become allocatable (10-15 min: driver compiles against the RHCOS kernel)"
for _ in $(seq 1 90); do
  TOTAL="$(oc get nodes -o json | jq '[.items[].status.allocatable["nvidia.com/gpu"] // "0" | tonumber] | add')"
  NODES="$(oc get nodes -o json | jq '[.items[] | select((.status.allocatable["nvidia.com/gpu"] // "0" | tonumber) > 0)] | length')"
  STATE="$(oc get clusterpolicy -o jsonpath='{.items[0].status.state}' 2>/dev/null || true)"
  echo "  $(date +%H:%M:%S) clusterpolicy=${STATE:-?} gpu nodes=${NODES} total gpus=${TOTAL}"
  [[ "$STATE" == "ready" && "$TOTAL" -gt 0 ]] && break
  sleep 20
done
oc get nodes -l nvidia.com/gpu.present=true -o custom-columns='NODE:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPUS:.status.allocatable.nvidia\.com/gpu,PRODUCT:.metadata.labels.nvidia\.com/gpu\.product'
