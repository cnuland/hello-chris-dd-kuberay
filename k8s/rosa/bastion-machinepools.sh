#!/usr/bin/env bash
# Run this ON THE RHDP BASTION (rosa CLI + AWS creds are preconfigured there and the cluster lives in
# RHDP's Red Hat org, so it is the only place `rosa create machinepool` works for this cluster).
#
#   scp k8s/rosa/bastion-machinepools.sh <user>@bastion.rrvnz.sandbox5285.opentlc.com:
#   ssh <user>@bastion.rrvnz.sandbox5285.opentlc.com ./bastion-machinepools.sh
#
# Adds the two pools the project needs:
#   cpu-workers  4 x m6i.8xlarge  (32 vCPU / 128 GiB each; ~100+ PyBoy emulators)   label workload=ray-cpu
#   gpu          4 x g6e.2xlarge  (1 x NVIDIA L40S 48 GB each; DiffusionGemma FP8)   label workload=gpu
set -euo pipefail
CLUSTER="${CLUSTER:-rosa-rrvnz}"
CPU_TYPE="${CPU_TYPE:-m6i.8xlarge}"; CPU_REPLICAS="${CPU_REPLICAS:-4}"
GPU_TYPE="${GPU_TYPE:-g6e.2xlarge}"; GPU_REPLICAS="${GPU_REPLICAS:-4}"

rosa whoami >/dev/null || { echo "rosa is not logged in on this host"; exit 1; }
echo "==> cluster"; rosa describe cluster -c "$CLUSTER" | grep -E "^(Name|State|Region|Control Plane|OpenShift Version|Availability Zones|Subnets)"
REGION="$(rosa describe cluster -c "$CLUSTER" -o json | jq -r '.region.id')"
SUBNET="$(rosa describe machinepool -c "$CLUSTER" --machinepool workers -o json 2>/dev/null | jq -r '.subnet // empty')"
[ -z "$SUBNET" ] && SUBNET="$(rosa list machinepools -c "$CLUSTER" -o json | jq -r '.[0].subnet // empty')"
echo "region=$REGION worker-subnet=${SUBNET:-<cluster default>}"
SUBNET_FLAG=(); [ -n "$SUBNET" ] && SUBNET_FLAG=(--subnet "$SUBNET")

if command -v aws >/dev/null; then
  echo "==> GPU vCPU quota (Running On-Demand G and VT instances) in $REGION:"
  aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region "$REGION" \
    --query 'Quota.Value' --output text 2>/dev/null || echo "  (could not read quota)"
  echo "   need $((GPU_REPLICAS * 8)) for ${GPU_REPLICAS} x ${GPU_TYPE}; request an increase if lower"
fi

ensure_pool() { # name type replicas labels
  if rosa describe machinepool -c "$CLUSTER" --machinepool "$1" >/dev/null 2>&1; then
    echo "==> machinepool $1 already exists"; return; fi
  echo "==> creating machinepool $1 ($3 x $2)"
  rosa create machinepool -c "$CLUSTER" --name "$1" --instance-type "$2" --replicas "$3" --labels "$4" \
    "${SUBNET_FLAG[@]}" --autorepair -y
}
ensure_pool cpu-workers "$CPU_TYPE" "$CPU_REPLICAS" "workload=ray-cpu"
ensure_pool gpu         "$GPU_TYPE" "$GPU_REPLICAS" "workload=gpu,nvidia.com/gpu.present=true"

echo "==> pools"; rosa list machinepools -c "$CLUSTER"
echo "Nodes take ~10 min to join. Watch with: oc get nodes -l workload -w"
