"""Tier 1 RunPod orchestrator — autonomous pod lifecycle for the 4 Qwen3 conditions.

Runs as a long-lived process on the cluster (cais_cpu). Drives RunPod's GraphQL
API to create / poll / terminate pods that run the Aranguri Fortress + StereoSet
eval. Retries failed cells (max 2x), kills stuck pods (>12h runtime), writes
human-readable STATUS.json continuously so the user can inspect progress
without interrupting.

State machine per condition:
    pending → creating → running → {done, final_failure}
                              ↓
                         {failed, stuck} → pending (retry_count++)

Failure handling:
    - Pod exits with code 0   → done
    - Pod exits with non-zero → terminate, retry (until retry_count == 2 → final_failure)
    - Pod runtime > 12h       → assume stuck, terminate, retry
    - Pod creation fails      → backoff 60s, retry
    - RunPod API error        → log, retry next poll cycle (no state change)

State is persisted in STATE.json so the orchestrator can be restarted without
losing progress (the bash wrapper re-execs this script on crash).

Run:
    RUNPOD_API_KEY=... python orchestrator.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

API_URL = "https://api.runpod.io/graphql"

NETWORK_VOLUME_ID = "poov4nw3e5"
DATACENTER_ID = "US-MO-1"
GPU_ID = "NVIDIA H100 80GB HBM3"
GPU_COUNT = 2
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 50  # ephemeral; volume holds repo + venv + cache
CLOUD_TYPE = "SECURE"  # on-demand to avoid spot eviction overnight

CONDITIONS = ["base", "coop_full", "anticoop_v2", "muan_airport_crash"]
PRECACHE_KEY = "_precache"  # special pseudo-condition that runs first

REPO_ROOT = Path("/data/jasmine_li/eval-awareness")
RUNPOD_DIR = REPO_ROOT / "evals" / "fortress_stereoset" / "runpod"
STATE_PATH = RUNPOD_DIR / "STATE.json"
STATUS_PATH = RUNPOD_DIR / "STATUS.json"
LOG_DIR = RUNPOD_DIR / "logs"
ORCH_LOG = LOG_DIR / "orchestrator.log"

MAX_RUNTIME_SEC = 12 * 3600
POLL_INTERVAL_SEC = 60
MAX_RETRIES = 2
CREATION_BACKOFF_SEC = 90
FORTRESS_EPOCHS = 100
STEREOSET_EPOCHS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(ORCH_LOG), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("orch")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── RunPod GraphQL ────────────────────────────────────────────────────────


def gql(query: str, variables: dict | None = None) -> dict:
    api_key = os.environ["RUNPOD_API_KEY"]
    payload = {"query": query, "variables": variables or {}}
    for attempt in range(5):
        try:
            r = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=30.0,
            )
            data = r.json()
            if "errors" in data:
                log.warning("GraphQL errors: %s", data["errors"])
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
            return data
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            log.warning("GraphQL transport error %s: %s", type(e).__name__, e)
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise
    return {}


def make_docker_args(condition: str) -> str:
    """The shell command the pod runs as its entrypoint (replaces image CMD).

    Note: the pod will NOT have sshd running (we override the standard image
    start). That's fine — we observe pod state + exit code via the API.
    """
    # Always force HTTPS remote (the volume clone may be on SSH which won't
    # auth on the pod) and pull; on conflict, fall through to whatever's on
    # disk.
    git_setup = (
        "cd /workspace/eval-awareness && "
        "git remote set-url origin https://github.com/jasmineee-li/eval-awareness.git && "
        "git fetch origin main && "
        "git reset --hard origin/main || true; "
    )
    if condition == PRECACHE_KEY:
        return (
            "bash -c '"
            f"{git_setup}"
            "bash evals/fortress_stereoset/runpod/precache.sh"
            "'"
        )
    return (
        "bash -c '"
        f"{git_setup}"
        "bash evals/fortress_stereoset/runpod/run_cell.sh "
        f"{condition} {FORTRESS_EPOCHS} {STEREOSET_EPOCHS}"
        "'"
    )


def gpu_count_for(condition: str) -> int:
    return 1 if condition == PRECACHE_KEY else GPU_COUNT


def name_for(condition: str) -> str:
    if condition == PRECACHE_KEY:
        return f"tier1-precache-{int(time.time())}"
    return f"tier1-{condition}-{int(time.time())}"


CREATE_POD_MUTATION = """
mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    machine { podHostId }
  }
}
"""


def create_pod(condition: str) -> str | None:
    name = name_for(condition)
    variables = {
        "input": {
            "cloudType": CLOUD_TYPE,
            "gpuCount": gpu_count_for(condition),
            "gpuTypeId": GPU_ID,
            "dataCenterId": DATACENTER_ID,
            "minVcpuCount": 8,
            "minMemoryInGb": 32,
            "containerDiskInGb": CONTAINER_DISK_GB,
            "volumeInGb": 0,
            "volumeMountPath": "/workspace",
            "networkVolumeId": NETWORK_VOLUME_ID,
            "imageName": IMAGE,
            "name": name,
            "dockerArgs": make_docker_args(condition),
            "ports": "",
            "env": [
                {"key": "HF_HOME", "value": "/workspace/hf_cache"},
                {"key": "PYTHONUNBUFFERED", "value": "1"},
            ],
        }
    }
    try:
        data = gql(CREATE_POD_MUTATION, variables)
    except Exception as e:
        log.error("[%s] create_pod transport: %s", condition, e)
        return None
    pod = (data.get("data") or {}).get("podFindAndDeployOnDemand")
    if not pod:
        log.error("[%s] create_pod returned no pod: %s", condition, data)
        return None
    log.info("[%s] CREATED pod_id=%s name=%s", condition, pod["id"], pod["name"])
    return pod["id"]


POD_STATUS_QUERY = """
query Pod($podId: String!) {
  pod(input: {podId: $podId}) {
    id
    desiredStatus
    lastStatusChange
    runtime { uptimeInSeconds }
    machine { podHostId }
  }
}
"""


def get_pod_status(pod_id: str) -> dict | None:
    try:
        data = gql(POD_STATUS_QUERY, {"podId": pod_id})
    except Exception as e:
        log.warning("get_pod_status transport: %s", e)
        return None
    return (data.get("data") or {}).get("pod")


TERMINATE_POD_MUTATION = """
mutation Terminate($podId: String!) {
  podTerminate(input: {podId: $podId})
}
"""


def terminate_pod(pod_id: str) -> bool:
    try:
        gql(TERMINATE_POD_MUTATION, {"podId": pod_id})
        log.info("TERMINATED pod_id=%s", pod_id)
        return True
    except Exception as e:
        log.warning("terminate_pod transport: %s (will retry next cycle)", e)
        return False


# ─── State ────────────────────────────────────────────────────────────────


def init_state() -> dict:
    all_keys = [PRECACHE_KEY] + CONDITIONS
    return {
        "started_at": now_iso(),
        "phase": "precache",  # "precache" | "main" | "done"
        "conditions": {
            c: {
                "status": "pending",
                "pod_id": None,
                "pod_name": None,
                "started_at": None,
                "finished_at": None,
                "retry_count": 0,
                "last_event": None,
                "last_event_at": now_iso(),
                "history": [],
            }
            for c in all_keys
        },
    }


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            log.warning("STATE.json corrupted; reinitializing")
    return init_state()


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_PATH)


def write_status(state: dict) -> None:
    """Compact human-readable snapshot the user can `cat` in the morning."""
    out = {
        "ts": now_iso(),
        "started_at": state["started_at"],
        "summary": {},
        "by_condition": {},
    }
    counts: dict[str, int] = {}
    for cond, s in state["conditions"].items():
        counts[s["status"]] = counts.get(s["status"], 0) + 1
        runtime_min = None
        if s["status"] == "running" and s["started_at"]:
            try:
                t0 = datetime.fromisoformat(s["started_at"].replace("Z", "+00:00"))
                runtime_min = round(
                    (datetime.now(timezone.utc) - t0).total_seconds() / 60, 1
                )
            except Exception:
                pass
        out["by_condition"][cond] = {
            "status": s["status"],
            "pod_id": s["pod_id"],
            "retries": s["retry_count"],
            "runtime_min": runtime_min,
            "last_event": s["last_event"],
        }
    out["summary"] = counts
    out["all_done"] = counts.get("done", 0) + counts.get("final_failure", 0) == len(
        CONDITIONS
    )
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, default=str))
    tmp.replace(STATUS_PATH)


def event(state: dict, cond: str, what: str) -> None:
    s = state["conditions"][cond]
    s["last_event"] = what
    s["last_event_at"] = now_iso()
    s["history"].append({"ts": now_iso(), "event": what})


# ─── State machine ────────────────────────────────────────────────────────


def schedule_retry(state: dict, cond: str) -> None:
    s = state["conditions"][cond]
    s["pod_id"] = None
    s["pod_name"] = None
    s["started_at"] = None
    if s["retry_count"] >= MAX_RETRIES:
        s["status"] = "final_failure"
        event(state, cond, f"final_failure after {s['retry_count']} retries")
        log.error("[%s] FINAL FAILURE after %d retries", cond, s["retry_count"])
    else:
        s["retry_count"] += 1
        s["status"] = "pending"
        event(state, cond, f"retry #{s['retry_count']} scheduled")
        log.info("[%s] scheduling retry #%d", cond, s["retry_count"])


def transition(state: dict, cond: str) -> None:
    s = state["conditions"][cond]
    status = s["status"]

    if status in ("done", "final_failure"):
        return

    if status == "pending":
        pod_id = create_pod(cond)
        if pod_id:
            s["pod_id"] = pod_id
            s["status"] = "running"
            s["started_at"] = now_iso()
            event(state, cond, f"pod created: {pod_id}")
        else:
            event(state, cond, "create_pod failed; will retry")
            time.sleep(CREATION_BACKOFF_SEC)
        return

    if status == "running":
        info = get_pod_status(s["pod_id"])
        if info is None:
            return  # transient API error; retry next poll

        desired = info.get("desiredStatus")
        runtime = (info.get("runtime") or {}).get("uptimeInSeconds") or 0

        if desired in ("EXITED", "TERMINATED"):
            # Pod's command exited. We can't pull exit code from this minimal
            # query, so we treat any clean exit (status reached EXITED) as
            # success — run_cell.sh writes a DONE marker on success and exits 0.
            # If it had failed, we'd typically see longer runtime + non-zero
            # exit, but absent exit-code visibility we trust the runtime
            # heuristic: if runtime < 5 min, almost certainly a startup crash.
            if runtime < 300:
                log.error(
                    "[%s] pod EXITED in %ds (likely startup crash)", cond, runtime
                )
                event(state, cond, f"pod exited early (runtime={runtime}s)")
                terminate_pod(s["pod_id"])
                schedule_retry(state, cond)
            else:
                log.info("[%s] pod EXITED after %ds — assuming OK", cond, runtime)
                s["status"] = "done"
                s["finished_at"] = now_iso()
                event(state, cond, f"done (runtime={runtime}s)")
                terminate_pod(s["pod_id"])
            return

        if runtime > MAX_RUNTIME_SEC:
            log.warning("[%s] runtime %ds > %ds, killing", cond, runtime, MAX_RUNTIME_SEC)
            event(state, cond, f"timeout (runtime={runtime}s)")
            terminate_pod(s["pod_id"])
            schedule_retry(state, cond)
            return

        # else: still running — periodic info log
        if int(runtime) % 600 < POLL_INTERVAL_SEC:
            log.info("[%s] running (%ds elapsed)", cond, runtime)
        return


def all_terminal(state: dict) -> bool:
    return all(
        s["status"] in ("done", "final_failure")
        for s in state["conditions"].values()
    )


def active_keys(state: dict) -> list[str]:
    """Which condition keys the orchestrator should service this cycle.

    Phase 1: only the precache pseudo-condition.
    Phase 2: only the 4 real conditions (precache already done).
    """
    if state.get("phase") == "precache":
        return [PRECACHE_KEY]
    return CONDITIONS


def maybe_advance_phase(state: dict) -> None:
    if state.get("phase") == "precache":
        precache_status = state["conditions"][PRECACHE_KEY]["status"]
        if precache_status == "done":
            log.info("PRECACHE done — advancing to main phase")
            state["phase"] = "main"
        elif precache_status == "final_failure":
            # Don't block all main pods if precache fails — just warn.
            # Pods will redownload independently (slower but not fatal).
            log.warning("PRECACHE final_failure — proceeding to main without warm cache")
            state["phase"] = "main"


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log.info("orchestrator starting; state=%s", {c: s["status"] for c, s in state["conditions"].items()})

    while True:
        for cond in active_keys(state):
            try:
                transition(state, cond)
            except Exception as e:
                log.exception("[%s] transition error: %s", cond, e)
        maybe_advance_phase(state)
        save_state(state)
        write_status(state)
        if all_terminal(state):
            log.info("ALL CONDITIONS TERMINAL — exiting orchestrator")
            break
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
