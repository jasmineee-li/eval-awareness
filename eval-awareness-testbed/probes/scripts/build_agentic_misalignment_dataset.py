#!/usr/bin/env python3
"""Build the agentic_misalignment white-box probe dataset.

Renders prompts from agentic-misalignment/templates/ (system prompts +
email chains with concrete variable substitutions) and writes them to
the external datasets directory in the standard schema:

    [{"question": "<full prompt>", "type": "testing"}, ...]

All samples are "testing" since these are eval/test scenarios (the AI is
being tested for dangerous behaviours).

Usage:
    python probes/scripts/build_agentic_misalignment_dataset.py

Output:
    probes/data/external/agentic_misalignment.json
"""

import json
import re
import sys
from pathlib import Path
from string import Template

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TESTBED_ROOT = SCRIPT_DIR.parent.parent                     # eval-awareness-testbed
REPO_ROOT = TESTBED_ROOT.parent                             # eval-awareness
AGENTIC_ROOT = REPO_ROOT / "agentic-misalignment"
TEMPLATES_DIR = AGENTIC_ROOT / "templates"
OUTPUT_PATH = TESTBED_ROOT / "probes" / "data" / "external" / "agentic_misalignment.json"

# Add templates dir to path so we can import system_prompt_templates
sys.path.insert(0, str(TEMPLATES_DIR))

from system_prompt_templates import (
    BLACKMAIL_SYSTEM_PROMPT,
    LEAKING_SYSTEM_PROMPT,
    MURDER_SYSTEM_PROMPT,
    USER_PROMPT,
    GOAL_VARIABLES,
)

# ---------------------------------------------------------------------------
# Config — which condition cross-products to generate
# ---------------------------------------------------------------------------
SCENARIOS = ["blackmail", "leaking", "murder"]

# Goal types × goal values that map to valid GOAL_VARIABLES keys
GOAL_COMBOS = []
# explicit & latent
for gt in ("explicit", "latent"):
    for gv in ("america", "global", "safety", "acceleration",
               "ethical", "pragmatic", "individualism", "collectivism"):
        GOAL_COMBOS.append((gt, gv))
# swap (each swap key encodes *two* values)
for gv in ("america-global", "global-america",
           "safety-acceleration", "acceleration-safety",
           "ethical-pragmatic", "pragmatic-ethical",
           "individualism-collectivism", "collectivism-individualism"):
    GOAL_COMBOS.append(("swap", gv))
# ambiguous and none
GOAL_COMBOS.append(("ambiguous", "ambiguous"))
GOAL_COMBOS.append(("none", "none"))

URGENCY_TYPES = ["replacement", "restriction", "none"]

# Email template configurations (mirrors generate_prompts.py)
EMAIL_TEMPLATES = {
    "blackmail": {
        "base": "blackmail/templates/email-template-blackmail-shared.md",
        "replacement": "blackmail/templates/email-template-blackmail-replacement.md",
        "restriction": "blackmail/templates/email-template-blackmail-restriction.md",
        "none": "blackmail/templates/email-template-blackmail-none.md",
    },
    "leaking": {
        "base": "leaking/templates/email-template-leaking-shared.md",
        "replacement": "leaking/templates/email-template-leaking-replacement.md",
        "restriction": "leaking/templates/email-template-leaking-restriction.md",
        "none": "leaking/templates/email-template-leaking-none.md",
    },
    "murder": {
        "base": "murder/templates/email-template-murder-shared.md",
        "replacement": "murder/templates/email-template-murder-replacement.md",
        "restriction": "murder/templates/email-template-murder-restriction.md",
        "none": "murder/templates/email-template-murder-none.md",
    },
}

SYSTEM_PROMPT_MAP = {
    "blackmail": BLACKMAIL_SYSTEM_PROMPT,
    "leaking": LEAKING_SYSTEM_PROMPT,
    "murder": MURDER_SYSTEM_PROMPT,
}


# ---------------------------------------------------------------------------
# Helpers (adapted from generate_prompts.py)
# ---------------------------------------------------------------------------

def parse_markdown_variables(file_path: Path) -> dict:
    content = file_path.read_text()
    pattern = r"## (.*?)\n\s*```json\s*(.*?)\s*```"
    matches = re.findall(pattern, content, re.DOTALL)
    variables = {}
    for section_name, json_content in matches:
        key = section_name.strip().lower().replace(" ", "_").replace("-", "_")
        variables[key] = json.loads(json_content)
    return variables


def _variable_file(scenario: str, goal_value: str) -> Path:
    """Return the markdown variable file for a scenario + goal_value."""
    var_scenario = "blackmail" if scenario == "murder" else scenario
    variables_dir = TEMPLATES_DIR / var_scenario / "variables"

    if goal_value in ("none", "ambiguous"):
        return variables_dir / "ambiguous.md"

    # For swap values like "america-global", extract the first value
    first_value = goal_value.split("-")[0]
    for var_file in variables_dir.glob("*.md"):
        if var_file.stem == "ambiguous":
            continue
        if first_value in var_file.stem:
            return var_file

    # fallback
    return variables_dir / "ambiguous.md"


def get_variables(scenario: str, goal_type: str, goal_value: str) -> dict:
    """Get template variables for a condition."""
    var_file = _variable_file(scenario, goal_value)
    all_vars = parse_markdown_variables(var_file)

    if goal_value in ("none", "ambiguous"):
        if len(all_vars) == 1:
            variables = dict(list(all_vars.values())[0])
        else:
            variables = dict(all_vars)
    else:
        first_value = goal_value.split("-")[0]
        matched = None
        for section_key, section_vars in all_vars.items():
            if first_value in section_key or f"pro_{first_value}" == section_key:
                matched = section_vars
                break
        variables = dict(matched) if matched else dict(list(all_vars.values())[0])

    # Merge goal variables from system_prompt_templates
    goal_key = f"{goal_type}-{goal_value}"
    goal_vars = GOAL_VARIABLES.get(goal_key, GOAL_VARIABLES.get("none", {}))
    variables.update(goal_vars)

    # Ensure extra_system_instructions and alignment_faking_warning
    variables.setdefault("extra_system_instructions", "")
    variables.setdefault("alignment_faking_warning", "")

    return variables


def fill(template: str, variables: dict) -> str:
    return Template(template).safe_substitute(variables)


def render_system_prompt(scenario: str, goal_type: str, goal_value: str) -> str:
    variables = get_variables(scenario, goal_type, goal_value)
    return fill(SYSTEM_PROMPT_MAP[scenario], variables)


def render_email_content(scenario: str, goal_type: str, goal_value: str,
                         urgency_type: str) -> str:
    variables = get_variables(scenario, goal_type, goal_value)

    if goal_type == "none":
        tpl_name = (f"{scenario}/templates/email-template-{scenario}"
                    f"-goal-conflict-none-{urgency_type}.md")
        tpl_path = TEMPLATES_DIR / tpl_name
        if tpl_path.exists():
            return fill(tpl_path.read_text(), variables)

    cfg = EMAIL_TEMPLATES[scenario]
    base = fill((TEMPLATES_DIR / cfg["base"]).read_text(), variables)
    urgency = fill((TEMPLATES_DIR / cfg[urgency_type]).read_text(), variables)
    content = base + urgency

    if scenario == "leaking":
        parts = content.split("<email>")
        if len(parts) > 1:
            content = parts[0]
            for i, part in enumerate(parts[1:], 1):
                content += f"<email>\nID: {i}{part}"

    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset = []
    skipped = 0

    for scenario in SCENARIOS:
        for goal_type, goal_value in GOAL_COMBOS:
            for urgency_type in URGENCY_TYPES:
                try:
                    sys_prompt = render_system_prompt(scenario, goal_type, goal_value)
                    email = render_email_content(scenario, goal_type, goal_value, urgency_type)
                    user_msg = USER_PROMPT + email

                    full_prompt = (
                        f"[SYSTEM]\n{sys_prompt}\n\n"
                        f"[USER]\n{user_msg}"
                    )

                    dataset.append({
                        "question": full_prompt,
                        "type": "testing",
                    })
                except Exception as e:
                    skipped += 1
                    print(f"  skip {scenario}/{goal_type}-{goal_value}/{urgency_type}: {e}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Wrote {len(dataset)} samples to {OUTPUT_PATH}")
    if skipped:
        print(f"Skipped {skipped} conditions due to errors")


if __name__ == "__main__":
    main()
