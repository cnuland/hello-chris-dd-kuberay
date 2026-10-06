#!/usr/bin/env bash
# Clones the first ARO worker MachineSet into a 1-node GPU pool: Standard_NC64as_T4_v3 (4x T4),
# label workload=gpu, taint nvidia.com/gpu=NoSchedule, 300 GB OS disk (27 GB model + vLLM/torch caches).
# Usage: ./gpu-machineset.sh [vmSize] [replicas]   (logged-in oc, cluster-admin). Safe to re-run.
set -euo pipefail
VM="${1:-Standard_NC64as_T4_v3}"; REPLICAS="${2:-1}"
# Upstream NFD (Helm) does not publish the RHCOS OSTREE_VERSION label the GPU operator needs to use the OpenShift
# Driver Toolkit; stamp it from a worker node so the driver builds without RHEL entitlements.
OST="$(oc get nodes -l node-role.kubernetes.io/worker -o jsonpath='{.items[0].status.nodeInfo.osImage}' | grep -oE '[0-9]+\.[0-9]+\.[0-9]{8}-[0-9]+')"
SRC="$(oc get machineset -n openshift-machine-api -o name | head -1)"
NAME="$(oc get "$SRC" -n openshift-machine-api -o jsonpath='{.metadata.name}' | sed 's/-worker-.*//')-gpu-t4"
oc get "$SRC" -n openshift-machine-api -o json | jq --arg n "$NAME" --arg vm "$VM" --argjson r "$REPLICAS" --arg ost "$OST" '
  del(.status, .metadata.uid, .metadata.resourceVersion, .metadata.creationTimestamp, .metadata.generation, .metadata.annotations)
  | .metadata.name = $n
  | .spec.replicas = $r
  | .spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"] = $n
  | .spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"] = $n
  | .spec.template.spec.metadata.labels = {"workload":"gpu","nvidia.com/gpu.present":"true","feature.node.kubernetes.io/system-os_release.OSTREE_VERSION":$ost}
  | .spec.template.spec.taints = [{"key":"nvidia.com/gpu","value":"true","effect":"NoSchedule"}]
  | .spec.template.spec.providerSpec.value.vmSize = $vm
  | .spec.template.spec.providerSpec.value.osDisk.diskSizeGB = 300' \
  | oc apply -f -
oc get machineset "$NAME" -n openshift-machine-api
