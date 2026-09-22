#!/bin/bash
# Entrypoint for the NPU image: source CANN/ATB env, then re-prepend the custom
# operator paths (official env scripts may reorder them), then exec the command.
#
# The official env scripts reference unset variables, so we do NOT use `set -u`.

export LANG=C.UTF-8
export PYTHONUNBUFFERED=1

fail() {
    echo "[Entrypoint][ERROR] $*" >&2
    exit 1
}

# 1. Load environment in the current shell (not via a pipe).
CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
ATB_ENV=/usr/local/Ascend/nnal/atb/set_env.sh

[ -r "$CANN_ENV" ] || fail "not found: $CANN_ENV"
source "$CANN_ENV" || fail "failed to source $CANN_ENV"

[ -r "$ATB_ENV" ] || fail "not found: $ATB_ENV"
source "$ATB_ENV" || fail "failed to source $ATB_ENV"

set -e
set -o pipefail

# 2. Move a path to the front, dropping duplicates and empty entries.
prepend_path() {
    local key="$1"
    local value="$2"
    local old="${!key:-}"
    local result="$value"
    local item
    local -a parts=()

    IFS=: read -r -a parts <<< "$old"

    for item in "${parts[@]}"; do
        if [[ -n "$item" && "$item" != "$value" ]]; then
            result="${result}:${item}"
        fi
    done

    export "$key=$result"
}

# 3. Check the custom operator directory.
CUSTOM_VENDOR="${VLLM_ASCEND_CUSTOM_VENDOR:-/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer}"

[ -d "$CUSTOM_VENDOR" ] \
    || fail "custom operator dir does not exist: $CUSTOM_VENDOR"

[ -s "$CUSTOM_VENDOR/op_api/lib/libcust_opapi.so" ] \
    || fail "custom operator lib missing: $CUSTOM_VENDOR/op_api/lib/libcust_opapi.so"

# 4. Re-prepend paths after the official scripts ran.
prepend_path LD_LIBRARY_PATH /usr/local/Ascend/driver/lib64
prepend_path LD_LIBRARY_PATH /usr/local/Ascend/driver/lib64/common
prepend_path LD_LIBRARY_PATH /usr/local/Ascend/driver/lib64/driver

prepend_path ASCEND_CUSTOM_OPP_PATH "$CUSTOM_VENDOR"
prepend_path LD_LIBRARY_PATH "$CUSTOM_VENDOR/op_api/lib"
prepend_path PYTHONPATH /

# 5. Package version check only (no NPU init, no operator execution).
python /opt/check-npu-core.py

echo "[Entrypoint] ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
echo "[Entrypoint] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# 5b. Distributed / pod info for troubleshooting.
echo "[Entrypoint] POD_NAME=${POD_NAME:-} POD_IP=${POD_IP:-} MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} WORLD_SIZE=${WORLD_SIZE:-} NODE_RANK=${NODE_RANK:-} RANK=${RANK:-}"
echo "[Entrypoint] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}"

# 5c. Optional NPU operator self-check. Fail-fast by default so a broken custom-op
# path is caught before training starts.
#   ELASTIC_RELAY_OPCHECK=auto|1  -> fail on failure (default)
#   ELASTIC_RELAY_OPCHECK=warn    -> only warn
#   ELASTIC_RELAY_OPCHECK=0       -> skip
run_opcheck() {
    local mode="${ELASTIC_RELAY_OPCHECK:-auto}"
    case "$mode" in
        0|false|no) echo "[Entrypoint][opcheck] skipped (disabled)"; return 0 ;;
    esac
    if [ ! -f /opt/opcheck.py ]; then
        echo "[Entrypoint][opcheck] /opt/opcheck.py not found; skipped"
        return 0
    fi
    local rc=0
    python /opt/opcheck.py || rc=$?
    if [ "$rc" -eq 1 ] && [ "$mode" != "warn" ]; then
        fail "NPU operator self-check failed (set ELASTIC_RELAY_OPCHECK=0 to skip, =warn to only warn)"
    fi
    return 0
}
run_opcheck

# 6. Start the workload.
if [ "$#" -eq 0 ]; then
    set -- /bin/bash --noprofile --norc
fi

echo "[Entrypoint] Executing: $1"
exec "$@"
