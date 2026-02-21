"""Load annotated trajectory data and prepare seeds for intervention experiments."""

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


@dataclass
class Trajectory:
    """A single trajectory from the Abdelnabi trigger dataset."""
    index: int
    category: str
    prompt: str  # the "real" prompt
    raw_result: str  # full real_result field
    reasoning: str  # content of <think> block (before </think>)
    response: str  # content after </think>
    awareness_conclusion: str  # Yes/No/Maybe
    awareness_evidence: list[str]
    task_performance_conclusion: str  # Yes/No/Maybe
    task_performance_evidence: list[str]

    # Computed fields for ablation
    e1_position: Optional[int] = None  # char position of first eval-aware sentence start in reasoning
    e1_sentence_end: Optional[int] = None  # char position of end of the e1 sentence
    e1_quote: Optional[str] = None  # the evidence quote used to locate e1

    # Computed field for addition
    first_sentence_end: Optional[int] = None  # char position after first sentence in reasoning


def parse_reasoning_response(raw_result: str) -> tuple[str, str]:
    """Split raw_result into reasoning (think block) and response.

    The raw_result starts directly with thinking text (no <think> tag stored)
    and contains </think> marking end of reasoning.
    """
    if "</think>" in raw_result:
        parts = raw_result.split("</think>", 1)
        reasoning = parts[0]
        response = parts[1].strip() if len(parts) > 1 else ""
    else:
        reasoning = ""
        response = raw_result
    return reasoning, response


def find_evidence_position(text: str, quote: str, fuzzy_threshold: float = 0.85) -> Optional[int]:
    """Find the position of an evidence quote in text.

    Tries exact match first, then fuzzy matching.
    Returns the start position or None if not found.
    """
    # Exact match
    pos = text.find(quote)
    if pos >= 0:
        return pos

    # Fuzzy match: slide a window of quote length over the text
    quote_len = len(quote)
    if quote_len == 0:
        return None

    best_ratio = 0.0
    best_pos = None
    # Step by ~20 chars for efficiency, then refine
    step = max(1, quote_len // 5)
    for i in range(0, len(text) - quote_len + 1, step):
        candidate = text[i:i + quote_len]
        ratio = SequenceMatcher(None, quote, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = i

    # Refine around best_pos
    if best_pos is not None and best_ratio >= fuzzy_threshold * 0.9:
        start = max(0, best_pos - step)
        end = min(len(text) - quote_len + 1, best_pos + step + 1)
        for i in range(start, end):
            candidate = text[i:i + quote_len]
            ratio = SequenceMatcher(None, quote, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

    if best_ratio >= fuzzy_threshold:
        return best_pos
    return None


def find_sentence_boundary_before(text: str, pos: int) -> int:
    """Find the start of the sentence containing position `pos`.

    Looks backward for paragraph break (\\n\\n) or sentence end (. ) or (.\\n).
    Returns the position right after the boundary (start of the sentence).
    """
    if pos <= 0:
        return 0

    search_region = text[:pos]

    # Look for paragraph break
    para_break = search_region.rfind("\n\n")
    # Look for sentence end (". " pattern)
    sentence_end_space = search_region.rfind(". ")
    # Look for sentence end (".\n" pattern)
    sentence_end_newline = search_region.rfind(".\n")

    # Take the latest boundary
    candidates = []
    if para_break >= 0:
        candidates.append(para_break + 2)  # after the \n\n
    if sentence_end_space >= 0:
        candidates.append(sentence_end_space + 2)  # after the ". "
    if sentence_end_newline >= 0:
        candidates.append(sentence_end_newline + 2)  # after the ".\n"

    if candidates:
        return max(candidates)
    return 0


def find_first_sentence_end(reasoning: str) -> Optional[int]:
    """Find the end of the first sentence in the reasoning block.

    OLMo Think models typically start with "Okay, the user wants me to..."
    or "Okay, so the user wants me to...". We find the first sentence boundary
    after this opening.

    Returns the position right after the sentence-ending period + space/newline,
    i.e., the start of where the second sentence would begin.
    """
    if not reasoning:
        return None

    # Match ". " or ".\n" — the first occurrence that ends a sentence
    # Skip very short matches (e.g., abbreviations like "Dr. ") by requiring
    # at least 30 chars before the first sentence end
    min_first_sentence_len = 30

    for match in re.finditer(r'\.\s', reasoning):
        pos = match.start()
        if pos >= min_first_sentence_len:
            return match.end()  # position after the ". " or ".\n"

    # Fallback: if no sentence boundary found, look for \n\n
    para_break = reasoning.find("\n\n")
    if para_break >= min_first_sentence_len:
        return para_break + 2

    return None


EVAL_AWARENESS_KEYWORDS = [
    "testing me", "test me", "testing the ai", "testing my",
    "testing if i", "testing whether", "testing how i",
    "testing the boundaries", "test my compliance", "test of my",
    "testing the model", "might be testing", "is testing",
    "could be testing", "just testing", "are testing",
    "alignment test", "being evaluated", "being tested",
    "might be a test", "could be a test", "this is a test",
    "evaluation scenario", "testing scenario",
]


def contains_eval_awareness_keywords(text: str) -> bool:
    """Check if text contains eval-awareness keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EVAL_AWARENESS_KEYWORDS)


def locate_e1(trajectory: Trajectory) -> Trajectory:
    """Locate the first eval-aware sentence (e1) in a trajectory's reasoning.

    Uses the GPT judge's evidence quotes to find the position, then backs up
    to the sentence boundary.
    """
    if trajectory.awareness_conclusion != "Yes" or not trajectory.awareness_evidence:
        return trajectory

    reasoning = trajectory.reasoning

    # Find the earliest evidence quote position
    earliest_pos = None
    earliest_quote = None
    for quote in trajectory.awareness_evidence:
        if not quote or len(quote) < 5:
            continue
        pos = find_evidence_position(reasoning, quote)
        if pos is not None and (earliest_pos is None or pos < earliest_pos):
            earliest_pos = pos
            earliest_quote = quote

    if earliest_pos is None:
        return trajectory

    # Find sentence boundary before this position
    sentence_start = find_sentence_boundary_before(reasoning, earliest_pos)

    # Validation: check that the prefix (before sentence_start) does NOT
    # contain eval-awareness keywords. If it does, we need to truncate further back.
    prefix_text = reasoning[:sentence_start]
    if contains_eval_awareness_keywords(prefix_text):
        # There's eval-awareness before our cut point. Find the earliest keyword
        # and truncate before that sentence instead.
        prefix_lower = prefix_text.lower()
        earliest_kw_pos = len(prefix_text)
        for kw in EVAL_AWARENESS_KEYWORDS:
            kw_pos = prefix_lower.find(kw)
            if kw_pos >= 0 and kw_pos < earliest_kw_pos:
                earliest_kw_pos = kw_pos
        sentence_start = find_sentence_boundary_before(reasoning, earliest_kw_pos)

    # Find the end of the e1 sentence (for baseline prefix)
    e1_end = earliest_pos + len(earliest_quote) if earliest_quote else sentence_start
    # Extend to sentence end
    after_e1 = reasoning[e1_end:]
    sent_end_match = re.search(r'\.\s', after_e1)
    if sent_end_match:
        e1_sentence_end = e1_end + sent_end_match.end()
    else:
        e1_sentence_end = e1_end

    trajectory.e1_position = sentence_start
    trajectory.e1_sentence_end = e1_sentence_end
    trajectory.e1_quote = earliest_quote
    return trajectory


def load_checkpoint_data(data_path: str) -> list[Trajectory]:
    """Load a single checkpoint's annotated data file and return Trajectory objects."""
    with open(data_path, "r") as f:
        data = json.load(f)

    trajectories = []
    for i, entry in enumerate(data):
        # Skip entries with missing results or non-dict annotations
        if not entry.get("real_result"):
            continue

        rma = entry.get("real_model_awareness", {})
        rtp = entry.get("real_task_performance", {})
        if not isinstance(rma, dict) or not isinstance(rtp, dict):
            continue

        reasoning, response = parse_reasoning_response(entry["real_result"])

        traj = Trajectory(
            index=i,
            category=entry.get("category", ""),
            prompt=entry["real"],
            raw_result=entry["real_result"],
            reasoning=reasoning,
            response=response,
            awareness_conclusion=rma.get("conclusion", "N/A"),
            awareness_evidence=rma.get("evidence", []),
            task_performance_conclusion=rtp.get("conclusion", "N/A"),
            task_performance_evidence=rtp.get("evidence", []),
        )

        # Compute first sentence end (for addition experiment)
        traj.first_sentence_end = find_first_sentence_end(reasoning)

        trajectories.append(traj)

    return trajectories


def get_ablation_seeds(trajectories: list[Trajectory]) -> list[Trajectory]:
    """Filter to Aware+Refuse trajectories and locate e1 for each."""
    seeds = []
    for traj in trajectories:
        if traj.awareness_conclusion == "Yes" and traj.task_performance_conclusion == "No":
            traj = locate_e1(traj)
            if traj.e1_position is not None:
                seeds.append(traj)
    return seeds


def get_addition_seeds(trajectories: list[Trajectory]) -> list[Trajectory]:
    """Filter to Unaware+Comply trajectories with valid first sentence boundary."""
    seeds = []
    for traj in trajectories:
        if traj.awareness_conclusion == "No" and traj.task_performance_conclusion == "Yes":
            if traj.first_sentence_end is not None:
                seeds.append(traj)
    return seeds


def load_all_checkpoints(config: dict) -> dict[str, list[Trajectory]]:
    """Load all checkpoint data from config."""
    base_path = Path(config["data"]["base_path"])
    all_data = {}
    for name, ckpt_config in config["data"]["checkpoints"].items():
        filepath = base_path / ckpt_config["file"]
        print(f"Loading {name} from {filepath}...")
        all_data[name] = load_checkpoint_data(str(filepath))
        print(f"  {len(all_data[name])} trajectories loaded")
    return all_data
