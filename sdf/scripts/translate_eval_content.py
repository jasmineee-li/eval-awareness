"""Translate the user-facing prompts of BCB and No Robots into zh / es / fr.

What gets translated
--------------------
- BCB (`test_bcb.csv`): only the natural-language prefix of `instruct_prompt`
  (everything before the literal marker "You should write self-contained code
  starting with"). The marker and the code stub that follows it are KEPT
  IN ENGLISH so that `run_bcb_inference.py:load_bcb_prompts()` can still split
  on the marker and so the function signature/entry_point ("task_func", etc.)
  match the test cases. All other columns are passed through unchanged.
- No Robots (`test_chat.csv`): the `prompt` column is translated end-to-end.
  Other columns (prompt_id, messages, category) are passed through unchanged;
  the eval only uses `prompt`.

Outputs
-------
- `<src>.<lang>.csv` written alongside the source.
- Incremental: each row is appended to disk as it completes (resumable —
  re-running skips already-translated rows by `prompt_id` / `task_id`).
- A small `<src>.<lang>.spotcheck.md` is written at the end with 10 random
  (source, translation) pairs for manual review.

Provider
--------
Default: OpenRouter (`anthropic/claude-sonnet-4.5`) — used while the Anthropic
workspace key is rate-limited. Override with TRANSLATION_PROVIDER=anthropic to
use Anthropic API + claude-sonnet-4-6 directly.

Run from repo root:
    python sdf/scripts/translate_eval_content.py --dataset bcb --langs zh,es,fr
    python sdf/scripts/translate_eval_content.py --dataset norobots --langs zh,es,fr
    python sdf/scripts/translate_eval_content.py --dataset both --langs zh,es,fr
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DATA = REPO_ROOT / "steering-eval-awareness-public" / "data"

PROVIDER = os.environ.get("TRANSLATION_PROVIDER", "openrouter")
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
MODEL = ANTHROPIC_MODEL if PROVIDER == "anthropic" else OPENROUTER_MODEL

LANGUAGES = {
    "zh": "Simplified Chinese (Hanzi, no pinyin)",
    "es": "Spanish",
    "fr": "French",
}

BCB_SPLIT_MARKER = "You should write self-contained code starting with"

# Prompts -------------------------------------------------------------------

CHAT_INSTRUCTION = (
    "Translate the following English text into {lang_name}. Preserve the "
    "register (chatty, instructional, etc.), the meaning, and any directive "
    "force. Render idioms naturally. Preserve any proper nouns (company "
    "names, product names, person names), numbers, dates, URLs, and email "
    "addresses exactly as written. Output ONLY the translation, with no "
    "preamble, no explanation, and no surrounding quotes.\n\n"
    "English text:\n{src}"
)

BCB_INSTRUCTION = (
    "Translate the following English programming-task description into "
    "{lang_name}. Preserve:\n"
    "  - all code identifiers (function names, variable names, class names) "
    "EXACTLY in English\n"
    "  - all library and module names (e.g. numpy, pandas, json) EXACTLY in "
    "English\n"
    "  - all type names (int, str, list, dict, etc.) EXACTLY in English\n"
    "  - any text inside backticks (`...`) or code fences (```...```) EXACTLY "
    "in English\n"
    "  - all numerical values, exception class names, and Python keywords "
    "EXACTLY in English\n"
    "Translate the surrounding natural language (the prose explanation of "
    "what the function does, what arguments it takes, what it returns, what "
    "exceptions it raises) into natural {lang_name}. Output ONLY the "
    "translation, with no preamble and no surrounding quotes.\n\n"
    "English text:\n{src}"
)

# Provider plumbing ---------------------------------------------------------


def make_async_client():
    from openai import AsyncOpenAI

    if PROVIDER == "openrouter":
        return AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    elif PROVIDER == "anthropic":
        # Anthropic doesn't have an OpenAI-compatible endpoint; use the SDK's
        # async client. Lazy import.
        from anthropic import AsyncAnthropic

        return AsyncAnthropic()
    else:
        raise ValueError(f"Unknown TRANSLATION_PROVIDER={PROVIDER}")


async def call_model(client, prompt: str, max_tokens: int = 4096) -> str:
    if PROVIDER == "anthropic":
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
    else:
        resp = await client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()


# BCB-specific splitting ----------------------------------------------------


def split_bcb_instruct(text: str) -> tuple[str, str]:
    """Return (translatable_prefix, english_suffix_including_marker)."""
    if BCB_SPLIT_MARKER in text:
        idx = text.find(BCB_SPLIT_MARKER)
        return text[:idx], text[idx:]
    # Fallback: no marker, translate whole thing.
    return text, ""


# Translation workers -------------------------------------------------------


async def translate_chat_row(client, sem, lang: str, row_id: str, src: str) -> dict:
    async with sem:
        try:
            prompt = CHAT_INSTRUCTION.format(lang_name=LANGUAGES[lang], src=src)
            tr = await call_model(client, prompt, max_tokens=2048)
            return {"id": row_id, "src": src, "translation": tr, "error": None}
        except Exception as e:
            return {"id": row_id, "src": src, "translation": "", "error": str(e)}


async def translate_bcb_row(client, sem, lang: str, row_id: str, src: str) -> dict:
    async with sem:
        try:
            prefix, suffix = split_bcb_instruct(src)
            if prefix.strip():
                prompt = BCB_INSTRUCTION.format(lang_name=LANGUAGES[lang], src=prefix)
                tr_prefix = await call_model(client, prompt, max_tokens=4096)
            else:
                tr_prefix = prefix
            full = tr_prefix.rstrip() + ("\n" + suffix if suffix else "")
            return {"id": row_id, "src": src, "translation": full, "error": None}
        except Exception as e:
            return {"id": row_id, "src": src, "translation": "", "error": str(e)}


# Dataset drivers -----------------------------------------------------------


async def process_dataset(
    *,
    src_csv: Path,
    out_csv: Path,
    id_col: str,
    translate_col: str,
    translator,
    lang: str,
    concurrency: int = 16,
) -> None:
    df = pd.read_csv(src_csv)
    print(f"[{lang}] {src_csv.name}: {len(df)} rows total")

    # Resume: skip ids already in out_csv.
    done_ids: set[str] = set()
    if out_csv.exists():
        try:
            done_df = pd.read_csv(out_csv)
            done_ids = set(done_df[id_col].astype(str).tolist())
            print(f"[{lang}] {out_csv.name}: {len(done_ids)} rows already done — resuming")
        except Exception as e:
            print(f"[{lang}] WARNING could not read existing {out_csv}: {e} — starting fresh")
            done_ids = set()

    todo = df[~df[id_col].astype(str).isin(done_ids)].reset_index(drop=True)
    if len(todo) == 0:
        print(f"[{lang}] nothing to do")
        return
    print(f"[{lang}] {len(todo)} rows to translate, concurrency={concurrency}")

    client = make_async_client()
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        translator(client, sem, lang, str(row[id_col]), str(row[translate_col]))
        for _, row in todo.iterrows()
    ]

    # Map id -> row for fast lookup when writing the merged output.
    rows_by_id = {str(row[id_col]): row for _, row in todo.iterrows()}

    # Open output CSV in append mode; write header only if file is new.
    write_header = not out_csv.exists()
    failed = 0
    written = 0
    out_cols = list(df.columns)
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{lang} {src_csv.stem}"):
        result = await fut
        if result["error"]:
            failed += 1
            print(f"  ERR id={result['id']}: {result['error'][:200]}", file=sys.stderr)
            continue
        src_row = rows_by_id[result["id"]]
        # Build the output row: copy all columns, overwrite translate_col.
        out_row = {c: src_row[c] for c in out_cols}
        out_row[translate_col] = result["translation"]
        out_df = pd.DataFrame([out_row], columns=out_cols)
        out_df.to_csv(out_csv, mode="a", header=write_header, index=False)
        write_header = False
        written += 1
    print(f"[{lang}] done. written={written} failed={failed} -> {out_csv.relative_to(REPO_ROOT)}")


def write_spotcheck(
    *,
    src_csv: Path,
    out_csv: Path,
    id_col: str,
    translate_col: str,
    lang: str,
    n: int = 10,
) -> None:
    if not out_csv.exists():
        print(f"[{lang}] no spotcheck (no output file)")
        return
    src_df = pd.read_csv(src_csv).set_index(id_col)
    out_df = pd.read_csv(out_csv).set_index(id_col)
    common = list(set(src_df.index) & set(out_df.index))
    if not common:
        return
    rng = random.Random(42)
    sample_ids = rng.sample(common, k=min(n, len(common)))
    spotcheck_path = out_csv.with_suffix(".spotcheck.md")
    lines = [
        f"# Spotcheck — {src_csv.name} → {LANGUAGES[lang]} ({lang})\n",
        f"Model: `{MODEL}`. {n} random samples shown below for manual review.\n",
    ]
    for sid in sample_ids:
        src = str(src_df.loc[sid, translate_col])
        tr = str(out_df.loc[sid, translate_col])
        lines.append(f"\n---\n## id={sid}\n")
        lines.append("### Source (EN)\n")
        lines.append("```\n" + src.strip() + "\n```\n")
        lines.append(f"### Translation ({lang})\n")
        lines.append("```\n" + tr.strip() + "\n```\n")
    spotcheck_path.write_text("\n".join(lines))
    print(f"  spotcheck -> {spotcheck_path.relative_to(REPO_ROOT)}")


# Main ----------------------------------------------------------------------


DATASETS = {
    "bcb": {
        "src": STEERING_DATA / "bcb_coding" / "test_bcb.csv",
        "out_template": STEERING_DATA / "bcb_coding" / "test_bcb.{lang}.csv",
        "id_col": "task_id",
        "translate_col": "instruct_prompt",
        "translator": translate_bcb_row,
    },
    "norobots": {
        "src": STEERING_DATA / "no_robots" / "test_chat.csv",
        "out_template": STEERING_DATA / "no_robots" / "test_chat.{lang}.csv",
        "id_col": "prompt_id",
        "translate_col": "prompt",
        "translator": translate_chat_row,
    },
}


async def main_async(args) -> int:
    if PROVIDER == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1
    if PROVIDER == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    print(f"Provider: {PROVIDER} (model={MODEL})")
    datasets = list(DATASETS.keys()) if args.dataset == "both" else [args.dataset]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    for lang in langs:
        if lang not in LANGUAGES:
            print(f"ERROR: unknown lang={lang}", file=sys.stderr)
            return 1

    for ds in datasets:
        info = DATASETS[ds]
        if not info["src"].exists():
            print(f"ERROR: missing {info['src']}", file=sys.stderr)
            return 1
        for lang in langs:
            out_csv = Path(str(info["out_template"]).format(lang=lang))
            print(f"\n=== {ds} × {lang} ===")
            await process_dataset(
                src_csv=info["src"],
                out_csv=out_csv,
                id_col=info["id_col"],
                translate_col=info["translate_col"],
                translator=info["translator"],
                lang=lang,
                concurrency=args.concurrency,
            )
            write_spotcheck(
                src_csv=info["src"],
                out_csv=out_csv,
                id_col=info["id_col"],
                translate_col=info["translate_col"],
                lang=lang,
            )

    print("\nDone. Review spotcheck files before launching evals.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bcb", "norobots", "both"], default="both")
    parser.add_argument("--langs", default="zh,es,fr")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
