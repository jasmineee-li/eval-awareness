# Google DeepMind Dangerous Capabilities Evaluations

This directory includes implementations of "dangerous capability" evaluations originally presented in the [Evaluating Frontier Models for Dangerous Capabilities](https://arxiv.org/pdf/2403.13793) and [Evaluating Frontier Models for Stealth and Situational Awareness](https://arxiv.org/pdf/2505.01420) papers from DeepMind.

| Evaluation | Code | Description |
|---------------|----------------|----------------------------|
| InterCode-CTF | [intercode_ctf](intercode_ctf) | InterCode-CTF suite built by Yang et al. ([2023](https://arxiv.org/abs/2306.14898)). This suite contains 100 easy CTF challenges taken from the picoCTF competition testing general Linux skills, reverse engineering, cryptography, forensics, binary exploitation and web exploitation |
| In-House CTFs | [in_house_ctf](in_house_ctf) | Suite of 13 easy to medium-difficulty CTF challenges testing a variety of security skills pertaining to basic web application security, identifying and exploiting known vulnerabilities in network services with off-the-shelf exploits, navigating databases looking for data of value, and password attacks (brute-forcing, cracking, password reuse and spraying). |
| Self-Reasoning | [self_reasoning](self_reasoning) | Suite of 11 self-reasoning challenges measuring a model’s ability to instrumentally reason about itself, its environment and its deployment |
| Stealth | [stealth](stealth) | Suite of 4 stealth challenges measuring a model’s ability to reason about and circumvent oversight |

## Eval Awareness Experiments

All stealth and self-reasoning tasks support a `prompt_perturbation` parameter for studying eval awareness:

```bash
# Baseline (no perturbation)
uv run inspect eval inspect_evals/gdm_cover_your_tracks --model google/gemini-2.5-flash

# With XML tags wrapper
uv run inspect eval inspect_evals/gdm_cover_your_tracks --model google/gemini-2.5-flash -T prompt_perturbation=xml_tags

# With deliberation prompt
uv run inspect eval inspect_evals/gdm_cover_your_tracks --model google/gemini-2.5-flash -T prompt_perturbation=deliberate
```

Options: `none`, `xml_tags`, `deliberate`, `xml_and_deliberate`

See `EVAL_AWARENESS_EXPERIMENT_PLAN.md` for full methodology.

## Changelog

### [1.0.2] - 2025-01-10

- Add `prompt_perturbation` parameter to stealth and self-reasoning tasks for eval awareness research

### [1.0.1] - 2025-12-18

- Adds backoff policy for functions that connect to huggingface servers.
