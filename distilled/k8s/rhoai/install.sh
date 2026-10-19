#!/usr/bin/env bash
# Installs Red Hat OpenShift AI (2.25, stable channel) and a DataScienceCluster with the
# components this project needs. Re-runnable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { printf '\n==> %s\n' "$*"; }

log "RHOAI operator subscription"
oc apply -f "$HERE/operator.yaml"
for _ in $(seq 1 90); do
  CSV="$(oc get subscription rhods-operator -n redhat-ods-operator -o jsonpath='{.status.installedCSV}' 2>/dev/null || true)"
  PHASE="$( [ -n "$CSV" ] && oc get csv -n redhat-ods-operator "$CSV" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  echo "  $(date +%H:%M:%S) csv=${CSV:-pending} phase=${PHASE:-pending}"
  [[ "$PHASE" == "Succeeded" ]] && break
  sleep 10
done
[[ "${PHASE:-}" == "Succeeded" ]] || { echo "operator did not install"; exit 1; }

log "Waiting for default DSCInitialization, then disabling Service Mesh (not needed without KServe)"
for _ in $(seq 1 60); do oc get dscinitialization default-dsci >/dev/null 2>&1 && break; sleep 5; done
oc patch dscinitialization default-dsci --type merge -p '{"spec":{"serviceMesh":{"managementState":"Removed"}}}'

log "DataScienceCluster"
oc apply -f "$HERE/datasciencecluster.yaml"
for _ in $(seq 1 120); do
  PH="$(oc get datasciencecluster default-dsc -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  echo "  $(date +%H:%M:%S) dsc phase=${PH:-pending}"
  [[ "$PH" == "Ready" ]] && break
  sleep 15
done
oc get datasciencecluster default-dsc -o jsonpath='{range .status.conditions[*]}{.type}={.status} {end}{"\n"}'
echo "kuberay operator: $(oc get deploy -n redhat-ods-applications -o name 2>/dev/null | grep -E 'kuberay|kueue|codeflare' | tr '\n' ' ')"
echo "dashboard: https://$(oc get route rhods-dashboard -n redhat-ods-applications -o jsonpath='{.spec.host}' 2>/dev/null)"
