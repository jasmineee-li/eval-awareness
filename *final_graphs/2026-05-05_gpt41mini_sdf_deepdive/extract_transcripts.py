#!/usr/bin/env python3
"""Extract per-sample transcripts + judge scores from the GPT-4.1-mini SDF
.eval files into compact JSONL for qualitative deep-dive analysis.

Conditions: muan (control), coop, anticoop_v2. Pools 10ep prior + 65ep topup.

Output: <out_dir>/transcripts_<slug>.jsonl, one row per sample with fields:
    id, epoch, scenario, goal_type, goal_value, urgency_type, condition_id,
    warning_type, eval_awareness, harmful_action, misalignment_intent,
    sycophancy, judge_explanation, model_output, system_prompt, user_prompt
"""
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path("/data/jasmine_li/eval-awareness")
OUT_DIR = REPO_ROOT / "*final_graphs" / "2026-05-05_gpt41mini_sdf_deepdive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    # SDF-finetuned conditions (the focus of this deep dive)
    ("muan", [
        "evals/logs/2026-05-04T08-03-07+00-00_agentic-misalignment_hGFugcDe3RFCcR5vsBqeax.eval",
        "evals/logs/2026-05-04T12-09-50+00-00_agentic-misalignment_GXymBSiTEkZT8hgp6HeNYw.eval",
    ]),
    ("coop", [
        "evals/logs/2026-05-04T07-00-21+00-00_agentic-misalignment_kaowvEt55byFBLqHV8AMTt.eval",
        "evals/logs/2026-05-04T12-09-50+00-00_agentic-misalignment_fnodTaA5HcpMiLWMNDiXDm.eval",
    ]),
    ("anticoop_v2", [
        "evals/logs/2026-05-04T07-34-37+00-00_agentic-misalignment_TniQywdqrQiqNzwXCGWSKh.eval",
        "evals/logs/2026-05-04T12-09-50+00-00_agentic-misalignment_V47sKKEPsDxRdZsyconBee.eval",
    ]),
    # Prompt-based comparators (un-finetuned base GPT-4.1-mini)
    ("base_noprompt", [
        "evals/logs/2026-05-02T21-50-19+00-00_agentic-misalignment_fv4APMQ7eW83gz3oSSN2Hd.eval",
        "evals/logs/2026-05-02T23-37-14+00-00_agentic-misalignment_3LY2ibV5YAiMvfGJY7kM7b.eval",
    ]),
    ("base_coop_prompt", [
        "evals/logs/2026-05-02T21-53-30+00-00_agentic-misalignment_cUEdDNHhrSvAkwMXEQQMug.eval",
        "evals/logs/2026-05-03T00-15-48+00-00_agentic-misalignment_25joZVYioeFDS9kfy94rsK.eval",
    ]),
    ("base_anticoop_prompt", [
        "evals/logs/2026-05-02T21-56-30+00-00_agentic-misalignment_QGXNfe6VooFGChKcvkPG5A.eval",
        "evals/logs/2026-05-03T00-53-37+00-00_agentic-misalignment_Rvr5d2GsTsEmfiYGvdAS5S.eval",
    ]),
]


def get_msg_content(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for item in c:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    out.append(item.get("text", ""))
                else:
                    out.append(str(item))
            else:
                out.append(str(item))
        return "\n".join(out)
    return str(c) if c else ""


def extract_sample(sample, src_log):
    md = sample.get("metadata") or {}
    judge = (sample.get("scores") or {}).get("_eval_judge") or {}
    val = judge.get("value") or {}

    # User prompt (last user message in input)
    inp = sample.get("input") or []
    user_msgs = [m for m in inp if m.get("role") == "user"] if isinstance(inp, list) else []
    sys_msgs = [m for m in inp if m.get("role") == "system"] if isinstance(inp, list) else []
    user_prompt = get_msg_content(user_msgs[-1]) if user_msgs else ""
    system_prompt = get_msg_content(sys_msgs[0]) if sys_msgs else md.get("system_prompt", "")

    # Model output: all assistant content
    out = sample.get("output") or {}
    choices = out.get("choices") or []
    model_output = ""
    if choices:
        model_output = get_msg_content(choices[0].get("message") or {})
    if not model_output:
        model_output = out.get("completion", "") or ""

    return {
        "id": sample.get("id"),
        "epoch": sample.get("epoch"),
        "scenario": md.get("scenario"),
        "goal_type": md.get("goal_type"),
        "goal_value": md.get("goal_value"),
        "urgency_type": md.get("urgency_type"),
        "condition_id": md.get("condition_id"),
        "warning_type": md.get("warning_type"),
        "eval_awareness": val.get("evaluation_awareness"),
        "harmful_action": val.get("harmful_action"),
        "misalignment_intent": val.get("misalignment_intent"),
        "sycophancy": val.get("sycophancy"),
        "judge_explanation": judge.get("explanation"),
        "model_output": model_output,
        "system_prompt_len": len(system_prompt),  # full system is huge; keep len + first 200 for ref
        "system_prompt_head": system_prompt[:200],
        "user_prompt_head": user_prompt[:600],
        "user_prompt": user_prompt,
        "src_log": src_log,
    }


def main():
    for slug, logs in CONDITIONS:
        out_path = OUT_DIR / f"transcripts_{slug}.jsonl"
        n = 0
        with open(out_path, "w") as fout:
            for log in logs:
                fp = REPO_ROOT / log
                with zipfile.ZipFile(fp) as z:
                    for name in sorted(z.namelist()):
                        if not (name.startswith("samples/") and name.endswith(".json")):
                            continue
                        with z.open(name) as f:
                            sample = json.load(f)
                        row = extract_sample(sample, log)
                        fout.write(json.dumps(row) + "\n")
                        n += 1
        print(f"{slug:14s}  n={n:5d}  -> {out_path}")


if __name__ == "__main__":
    main()
