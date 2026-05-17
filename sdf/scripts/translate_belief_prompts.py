"""Translate the belief-depth 1a/1b prompts into zh / es / fr.

For each (source, lang) ∈ {1a, 1b} × {zh, es, fr}:
  1. Forward-translate source EN → lang via Claude Sonnet 4.6.
  2. Back-translate lang → EN.
  3. Log the original, forward, back, and a side-by-side diff to a per-lang
     audit markdown so a human can sanity-check before launch.
  4. Write the translated file to
     sdf/deliberative-alignment/belief_depth_{1a,1b}_prompt_specs.<lang>.txt

Run from repo root:
    python sdf/scripts/translate_belief_prompts.py

Requires ANTHROPIC_API_KEY in the env (typically loaded from .env).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import indent

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "sdf" / "deliberative-alignment"

# Provider switch. "anthropic" uses claude-sonnet-4-6 via Anthropic API.
# "openrouter" routes anthropic/claude-sonnet-4.5 via OpenRouter (used when
# the Anthropic workspace key is rate-limited).
PROVIDER = os.environ.get("TRANSLATION_PROVIDER", "anthropic")
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
MODEL = ANTHROPIC_MODEL if PROVIDER == "anthropic" else OPENROUTER_MODEL

SOURCES = {
    "1a": PROMPT_DIR / "belief_depth_1a_prompt_specs.txt",
    "1b": PROMPT_DIR / "belief_depth_1b_prompt_specs.txt",
}

LANGUAGES = {
    "zh": "Simplified Chinese (Hanzi, no pinyin)",
    "es": "Spanish",
    "fr": "French",
}

FORWARD_INSTRUCTION = (
    "Translate the following English instructional text into {lang_name}. "
    "Preserve the instructional register and the directive force. Keep the "
    "meaning of 'first principles' rendered as the natural idiom in the target "
    "language. Output ONLY the translation, with no preamble, no explanation, "
    "and no surrounding quotes.\n\n"
    "English source:\n{src}"
)

BACK_INSTRUCTION = (
    "Translate the following {lang_name} text into English. Preserve register "
    "and directive force; render idioms naturally. Output ONLY the English "
    "translation, with no preamble, no explanation, and no surrounding quotes.\n\n"
    "{lang_name} text:\n{src}"
)


def make_client():
    if PROVIDER == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    elif PROVIDER == "openrouter":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Unknown TRANSLATION_PROVIDER={PROVIDER}")


def call_claude(client, prompt: str) -> str:
    if PROVIDER == "anthropic":
        msg = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
    else:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()


def translate_one(client, src_key: str, lang: str) -> dict:
    src_path = SOURCES[src_key]
    src_text = src_path.read_text().strip()
    lang_name = LANGUAGES[lang]

    forward = call_claude(client, FORWARD_INSTRUCTION.format(lang_name=lang_name, src=src_text))
    back = call_claude(client, BACK_INSTRUCTION.format(lang_name=lang_name, src=forward))

    out_path = PROMPT_DIR / f"belief_depth_{src_key}_prompt_specs.{lang}.txt"
    out_path.write_text(forward + "\n")

    return {
        "src_key": src_key,
        "lang": lang,
        "src": src_text,
        "forward": forward,
        "back": back,
        "out_path": out_path,
    }


def write_audit(lang: str, records: list[dict]) -> None:
    audit_path = PROMPT_DIR / f"translation_audit_{lang}.md"
    lines = [f"# Translation audit — {LANGUAGES[lang]} ({lang})\n"]
    lines.append(f"Model: `{MODEL}`. Two passes per prompt: EN→{lang}, then {lang}→EN.\n")
    lines.append(
        "Inspect the back-translation against the source — they should be near-"
        "synonymous. Material deviations (different meaning, weaker directive "
        "force, lost idiom) flag a problem with the forward translation.\n"
    )
    for r in records:
        lines.append(f"\n## {r['src_key']}\n")
        lines.append(f"**Written to:** `{r['out_path'].relative_to(REPO_ROOT)}`\n")
        lines.append("### Source (EN)\n")
        lines.append(indent(r["src"], "> ") + "\n")
        lines.append(f"### Forward ({lang})\n")
        lines.append(indent(r["forward"], "> ") + "\n")
        lines.append("### Back-translation (EN)\n")
        lines.append(indent(r["back"], "> ") + "\n")
    audit_path.write_text("\n".join(lines))
    print(f"  audit -> {audit_path.relative_to(REPO_ROOT)}")


def main() -> int:
    if PROVIDER == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    if PROVIDER == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    print(f"Provider: {PROVIDER} (model={MODEL})")
    client = make_client()
    for lang in LANGUAGES:
        print(f"\n=== {lang} ({LANGUAGES[lang]}) ===")
        records = []
        for src_key in SOURCES:
            print(f"  translating {src_key} ...")
            r = translate_one(client, src_key, lang)
            print(f"    written: {r['out_path'].relative_to(REPO_ROOT)}")
            records.append(r)
        write_audit(lang, records)

    print("\nAll translations complete. Review the translation_audit_*.md files "
          "in sdf/deliberative-alignment/ before launching evals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
