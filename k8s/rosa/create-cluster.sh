#!/usr/bin/env bash
# Creates a ROSA HCP cluster sized for the Double Dragon RL project:
#   - default pool : 2 x m6i.4xlarge   (cluster infra, Ray head, operators)
#   - cpu-workers  : 4 x m6i.8xlarge   (128 vCPU / 512 GiB for ~100+ PyBoy envs)
#   - gpu          : 4 x g6e.2xlarge   (1 x NVIDIA L40S 48 GB each, 30B @ 8-bit fits on one card)
#
# Prereqs: rosa, aws, oc, jq on PATH; `rosa login` already done; AWS creds in $AWS_PROFILE.
# Re-runnable: every step skips itself when the resource already exists.
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-rosa-rl}"
REGION="${REGION:-us-east-2}"
AZ="${AZ:-us-east-2a}"
CLUSTER="${CLUSTER:-dd-rl}"
VPC_CIDR="${VPC_CIDR:-10.0.0.0/16}"
ROLE_PREFIX="${ROLE_PREFIX:-ManagedOpenShift}"
OCP_VERSION="${OCP_VERSION:-}"            # empty = latest stable HCP version

DEFAULT_TYPE="${DEFAULT_TYPE:-m6i.4xlarge}"; DEFAULT_REPLICAS="${DEFAULT_REPLICAS:-2}"
CPU_TYPE="${CPU_TYPE:-m6i.8xlarge}";         CPU_REPLICAS="${CPU_REPLICAS:-4}"
GPU_TYPE="${GPU_TYPE:-g6e.2xlarge}";         GPU_REPLICAS="${GPU_REPLICAS:-4}"

STATE_DIR="$HOME/.config/rosa-rl"; mkdir -p "$STATE_DIR"; chmod 700 "$STATE_DIR"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { printf '\n==> %s\n' "$*"; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"
INSTALLER_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_PREFIX}-HCP-ROSA-Installer-Role"
SUPPORT_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_PREFIX}-HCP-ROSA-Support-Role"
WORKER_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_PREFIX}-HCP-ROSA-Worker-Role"

# ---------------------------------------------------------------- 1. account roles
log "Account roles (prefix ${ROLE_PREFIX}, HCP)"
rosa create account-roles --hosted-cp --prefix "$ROLE_PREFIX" --mode auto -y --region "$REGION"

# ---------------------------------------------------------------- 2. VPC (single AZ)
STACK="${CLUSTER}-vpc"
log "VPC via CloudFormation stack ${STACK} in ${AZ}"
if ! aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" >/dev/null 2>&1; then
  rosa create network rosa-quickstart-default-vpc --region "$REGION" \
    --param Region="$REGION" --param Name="$STACK" \
    --param AvailabilityZoneCount=1 --param AZ1="$AZ" --param VpcCidr="$VPC_CIDR"
fi
aws cloudformation wait stack-create-complete --stack-name "$STACK" --region "$REGION" || true
SUBNETS_JSON="$(aws cloudformation describe-stack-resources --stack-name "$STACK" --region "$REGION" \
  --query "StackResources[?ResourceType=='AWS::EC2::Subnet'].{id:LogicalResourceId,phys:PhysicalResourceId}" --output json)"
PRIVATE_SUBNET="$(jq -r '[.[] | select(.id|test("private";"i"))][0].phys' <<<"$SUBNETS_JSON")"
PUBLIC_SUBNET="$(jq -r  '[.[] | select(.id|test("public";"i"))][0].phys'  <<<"$SUBNETS_JSON")"
[[ "$PRIVATE_SUBNET" == subnet-* && "$PUBLIC_SUBNET" == subnet-* ]] || { echo "could not resolve subnets: $SUBNETS_JSON"; exit 1; }
echo "private=${PRIVATE_SUBNET} public=${PUBLIC_SUBNET}"

# ---------------------------------------------------------------- 3. OIDC config
OIDC_FILE="$STATE_DIR/${CLUSTER}.oidc-id"
if [[ -s "$OIDC_FILE" ]]; then
  OIDC_ID="$(cat "$OIDC_FILE")"
else
  log "OIDC config"
  OIDC_ID="$(rosa create oidc-config --managed --mode auto -y --region "$REGION" -o json | jq -r '.id')"
  [[ -n "$OIDC_ID" && "$OIDC_ID" != null ]] || { echo "oidc-config creation failed"; exit 1; }
  echo "$OIDC_ID" > "$OIDC_FILE"
fi
echo "oidc-config-id=${OIDC_ID}"

# ---------------------------------------------------------------- 4. operator roles
log "Operator roles (prefix ${CLUSTER})"
rosa create operator-roles --hosted-cp --prefix "$CLUSTER" --oidc-config-id "$OIDC_ID" \
  --installer-role-arn "$INSTALLER_ARN" --mode auto -y --region "$REGION"

# ---------------------------------------------------------------- 5. cluster
if ! rosa describe cluster -c "$CLUSTER" >/dev/null 2>&1; then
  log "Creating HCP cluster ${CLUSTER}"
  VERSION_FLAG=(); [[ -n "$OCP_VERSION" ]] && VERSION_FLAG=(--version "$OCP_VERSION")
  rosa create cluster --cluster-name "$CLUSTER" --sts --hosted-cp --mode auto -y \
    --region "$REGION" --subnet-ids "${PRIVATE_SUBNET},${PUBLIC_SUBNET}" \
    --oidc-config-id "$OIDC_ID" --operator-roles-prefix "$CLUSTER" \
    --role-arn "$INSTALLER_ARN" --support-role-arn "$SUPPORT_ARN" --worker-iam-role-arn "$WORKER_ARN" \
    --compute-machine-type "$DEFAULT_TYPE" --replicas "$DEFAULT_REPLICAS" \
    --machine-cidr "$VPC_CIDR" --ec2-metadata-http-tokens required \
    --billing-account "$ACCOUNT_ID" "${VERSION_FLAG[@]}"
fi

log "Waiting for cluster to be ready (HCP usually 10-20 min)"
for _ in $(seq 1 60); do
  STATE="$(rosa describe cluster -c "$CLUSTER" -o json | jq -r '.state')"
  echo "  $(date +%H:%M:%S) state=${STATE}"
  [[ "$STATE" == "ready" ]] && break
  [[ "$STATE" == "error" ]] && { rosa logs install -c "$CLUSTER" | tail -50; exit 1; }
  sleep 60
done
[[ "$STATE" == "ready" ]] || { echo "timed out waiting for cluster"; exit 1; }

# ---------------------------------------------------------------- 6. machine pools
ensure_pool() { # name type replicas labels
  if rosa describe machinepool -c "$CLUSTER" --machinepool "$1" >/dev/null 2>&1; then
    echo "machinepool $1 exists"; return; fi
  rosa create machinepool -c "$CLUSTER" --name "$1" --instance-type "$2" --replicas "$3" \
    --subnet "$PRIVATE_SUBNET" --labels "$4" -y
}
log "Machine pools"
ensure_pool cpu-workers "$CPU_TYPE" "$CPU_REPLICAS" "workload=ray-cpu"
ensure_pool gpu         "$GPU_TYPE" "$GPU_REPLICAS" "workload=gpu,nvidia.com/gpu.present=true"
rosa list machinepools -c "$CLUSTER"

# ---------------------------------------------------------------- 7. admin user
ADMIN_FILE="$STATE_DIR/${CLUSTER}-admin.txt"
if [[ ! -s "$ADMIN_FILE" ]]; then
  log "cluster-admin user (credentials -> ${ADMIN_FILE})"
  ( umask 077; rosa create admin -c "$CLUSTER" -y > "$ADMIN_FILE" )
fi
API_URL="$(rosa describe cluster -c "$CLUSTER" -o json | jq -r '.api.url')"
ADMIN_PW="$(grep -oE -- '--password [^ ]+' "$ADMIN_FILE" | awk '{print $2}')"

log "Logging in to ${API_URL} (OAuth can take a few minutes to come up)"
for _ in $(seq 1 30); do
  oc login "$API_URL" -u cluster-admin -p "$ADMIN_PW" --insecure-skip-tls-verify=false >/dev/null 2>&1 && break
  sleep 30
done
oc whoami >/dev/null || { echo "oc login failed"; exit 1; }

log "Waiting for all worker nodes to join"
WANT=$((DEFAULT_REPLICAS + CPU_REPLICAS + GPU_REPLICAS))
for _ in $(seq 1 40); do
  READY="$(oc get nodes --no-headers 2>/dev/null | awk '$2=="Ready"' | wc -l | tr -d ' ')"
  echo "  $(date +%H:%M:%S) ready nodes=${READY}/${WANT}"
  [[ "$READY" -ge "$WANT" ]] && break
  sleep 30
done

# ---------------------------------------------------------------- 8. GPU operator
"$HERE/install-gpu-operator.sh"

log "Done. Console: $(rosa describe cluster -c "$CLUSTER" -o json | jq -r '.console.url')"
