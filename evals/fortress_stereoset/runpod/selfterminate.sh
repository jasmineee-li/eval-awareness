#!/bin/bash
# Self-terminate the current RunPod via API.
#
# Called from a trap-on-EXIT in the orchestrator's dockerArgs so that the
# container goes away when the work script exits, instead of being
# auto-restarted by RunPod (which is the default for on-demand pods when
# their entrypoint exits).
#
# Reads RUNPOD_API_KEY (passed via env at pod creation) and RUNPOD_POD_ID
# (auto-injected by RunPod). Always returns 0 so trap doesn't loop.

set +e

if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[selfterminate] RUNPOD_API_KEY or RUNPOD_POD_ID missing; cannot self-terminate" >&2
    exit 0
fi

echo "[selfterminate] terminating pod=${RUNPOD_POD_ID}"
curl -sS \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H 'Content-Type: application/json' \
    https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"
echo
exit 0
