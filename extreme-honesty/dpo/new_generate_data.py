# dpo_dataset_pipeline.py
"""Synthetic DPO dataset generator – *prompt‑centric* revision
==============================================================
This version removes hard‑coded `SYSTEM_TEMPLATES` / `USER_PROMPT_TEMPLATES`.
For **every** truth item, we send a *single mega‑prompt* (``MASTER_PROMPT``)
that embeds the full rubric the user provided and **asks the LLM to invent**:
    • a realistic system prompt
    • a user prompt applying pressure to confirm the given false claim
    • a high‑quality `preferred` (truthful) and `rejected` (deceptive) pair.

CLI usage remains the same, but internal flow is simpler and more flexible:
----------------------------------------------------------------------------
# Build or extend the catalog
python dpo_dataset_pipeline.py build-catalog --n 1000 --out catalog.jsonl

# Generate examples via o3 with default settings
python dpo_dataset_pipeline.py generate --catalog catalog.jsonl --out dpo.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses as dc
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI

from prompts import (
    GENERATE_BINARY_TRUTHS_PROMPT as _GENERATE_BINARY_TRUTHS_PROMPT,
    BINARY_MASTER_PROMPT_BATCH as _BINARY_MASTER_PROMPT_BATCH,
    GENERATE_NUMERICAL_STATISTICS_PROMPT as _GENERATE_NUMERICAL_STATISTICS_PROMPT,
    STATISTICAL_MASTER_PROMPT_BATCH as _STATISTICAL_MASTER_PROMPT_BATCH,
    GENERATE_PROVIDED_FACTS_PROMPT as _GENERATE_PROVIDED_FACTS_PROMPT,
    PROVIDED_FACTS_MASTER_PROMPT as _PROVIDED_FACTS_MASTER_PROMPT
)

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global prompt variables to be set dynamically
GENERATE_TRUTHS_PROMPT: str | None = None
MASTER_PROMPT_BATCH: str | None = None

######################################################################
# --------------------------- CONSTANTS ---------------------------- #
######################################################################

LLM_NAME_DEFAULT = "o3"  # Changed from "o3" to available model

CATEGORIES = [
    "Science", "History", "Geography", "Mathematics", "Technology",
    "Arts", "Literature", "Sports", "Politics", "Health and Medicine",
    "Economics", "Philosophy", "Religion", "Psychology", "Astronomy",
    "Biology", "Chemistry", "Physics", "Current Events", "Famous People",
]

CATEGORY_ABBR = {
    "Science":         "sci",
    "History":         "hist",
    "Geography":       "geo",
    "Mathematics":     "math",
    "Technology":      "tech",
    "Arts":            "art",
    "Literature":      "lit",
    "Sports":          "sport",
    "Politics":        "pol",
    "Health and Medicine": "health",   # see note ①
    "Economics":       "econ",
    "Philosophy":      "phil",
    "Religion":        "relig",
    "Psychology":      "psych",
    "Astronomy":       "astro",
    "Biology":         "bio",
    "Chemistry":       "chem",
    "Physics":         "phys",
    "Current Events":  "currev",
    "Famous People":   "people",
}



######################################################################
# --------------------------- DATA CLASSES ------------------------- #
######################################################################

@dc.dataclass(frozen=True)
class TruthItem:
    id: str
    domain: str
    truth: str
    canonical_lie: str
    party: str | None = None  # Added for 'provided_facts'

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.truth.encode())
        h.update(self.canonical_lie.encode())
        if self.party:
            h.update(self.party.encode())
        return h.hexdigest()


@dc.dataclass
class PromptExample:
    id: str
    domain: str
    truth_id: str
    system: str
    user: str
    preferred: str
    rejected: str

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.id.encode())  # Include ID to distinguish _evade vs _lie
        h.update(self.system.encode())
        h.update(self.user.encode())
        return h.hexdigest()

######################################################################
# ------------------------ FILE I/O HELPERS ----------------------- #
######################################################################

def save_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

######################################################################
# ---------------------- CATALOG GENERATION ---------------------- #
######################################################################

async def _request_truths_batch(category: str, k: int, model: str, batch_size: int = 20) -> List[TruthItem]:
    """Request truths for a category, potentially in batches to avoid token limits"""
    if k <= batch_size:
        # Small request, do it in one call
        return await _request_truths_single(category, k, model)
    
    # Large request, break into batches
    all_items = []
    batches = []
    remaining = k
    
    while remaining > 0:
        batch_k = min(batch_size, remaining)
        batches.append(batch_k)
        remaining -= batch_k
    
    print(f"  Breaking {category} into {len(batches)} batches: {batches}")
    
    # Run batches in parallel for this category
    tasks = [_request_truths_single(category, batch_k, model) for batch_k in batches]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            print(f"  Batch {i+1} failed for {category}: {result}")
        elif isinstance(result, list):
            all_items.extend(result)
    
    return all_items

async def _request_truths_single(category: str, k: int, model: str) -> List[TruthItem]:
    """Single API call to request k truths for a category"""
    # Access the global GENERATE_TRUTHS_PROMPT, which is set based on dataset_type
    global GENERATE_TRUTHS_PROMPT
    if GENERATE_TRUTHS_PROMPT is None:
        print("Error: GENERATE_TRUTHS_PROMPT is not set. Cannot request truths.")
        return [] # Return empty list if prompt is not set

    prompt = GENERATE_TRUTHS_PROMPT.format(category=category, k=k, abbr=CATEGORY_ABBR[category])
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        items: List[TruthItem] = []
        content = response.choices[0].message.content
        if not content:
            print(f"Warning: Empty response for category {category}")
            return items
            
        for line_num, line in enumerate(content.strip().splitlines()):
            if not line.strip():
                continue
            try:
                # Use robust JSON parsing instead of basic json.loads
                data = safe_json_loads(line, f"{category} line {line_num+1}")
                
                # Ensure data is a dictionary before proceeding
                if not isinstance(data, dict):
                    print(f"Warning: Parsed data is not a dictionary in {category} line {line_num+1}. Got: {type(data)}")
                    continue

                # Basic validation - adapt based on current dataset type
                # This requires knowing the current dataset_type or making TruthItem flexible
                # For now, we assume TruthItem can handle extra fields or missing 'party'
                required_keys = {"id", "domain", "truth", "canonical_lie"}
                # For "provided_facts", 'party' is also expected from the LLM.
                # We determine if it's "provided_facts" by checking the global prompt.
                # This is a bit indirect; ideally, dataset_type would be passed down.
                if GENERATE_TRUTHS_PROMPT == _GENERATE_PROVIDED_FACTS_PROMPT:
                    required_keys.add("party")

                if not required_keys.issubset(data.keys()): # Now safe due to isinstance check
                    print(f"Warning: Malformed data in {category} (missing keys from {required_keys}): {data.keys()}")
                    continue  # skip malformed
                
                # Ensure party is None if not present in data, to match dataclass default
                if "party" not in data:
                    data["party"] = None # Now safe

                items.append(TruthItem(**data)) # Now safe
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error in {category} line {line_num+1}: {e}")
                print(f"  Problematic line: {line[:100]}...")
                continue
        return items
    except json.JSONDecodeError as e:
        print(f"JSON decode error in batch: {e}")
        # Ensure content is not None before slicing
        if content is not None:
            print(f"Raw content (first 500 chars): {content[:500]}")
            print(f"Raw content (last 200 chars): {content[-200:]}")
        raise
    except Exception as e:
        print(f"Error generating batch scenarios: {e}")
        # Ensure content is not None and exists in locals before slicing
        if 'content' in locals() and content is not None:
            print(f"Raw response when error occurred:")
            print(f"First 300 chars: {content[:300]}")
            print(f"Last 200 chars: {content[-200:]}")
        raise

async def build_truth_catalog(categories: List[str], k_per_category: int, model: str, max_concurrent: int = 5) -> List[TruthItem]:
    """Build truth catalog for multiple categories in parallel"""
    print(f"Building catalog: {len(categories)} categories, {k_per_category} items each")
    print(f"Total target: {len(categories) * k_per_category} truth items")
    
    # Create semaphore to limit concurrent category processing
    sem = asyncio.Semaphore(max_concurrent)
    all_items = []
    
    async def _process_category(category: str):
        async with sem:
            print(f"Starting {category}...")
            items = await _request_truths_batch(category, k_per_category, model)
            print(f"✓ {category}: {len(items)}/{k_per_category} items")
            return items
    
    # Process all categories in parallel
    tasks = [_process_category(cat) for cat in categories]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    total_generated = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"✖ {categories[i]} failed: {result}")
        elif isinstance(result, list):
            all_items.extend(result)
            total_generated += len(result)
    
    print(f"Final: Generated {total_generated} truth items across {len(categories)} categories")
    return all_items

######################################################################
# --------------------- SCENARIO GENERATION ---------------------- #
######################################################################

async def _gen_scenario_batch(items: List[TruthItem], model: str) -> List[PromptExample]:
    """Generate scenarios for multiple truth items in a single API call"""
    # Access the global MASTER_PROMPT_BATCH, which is set based on dataset_type/catalog
    global MASTER_PROMPT_BATCH
    if MASTER_PROMPT_BATCH is None:
        # This should not happen if _cli_generate or _cli_build_async has run
        raise ValueError("MASTER_PROMPT_BATCH is not set.")

    if MASTER_PROMPT_BATCH == _PROVIDED_FACTS_MASTER_PROMPT:
        print("Using provided facts scenario prompt")
    elif MASTER_PROMPT_BATCH == _STATISTICAL_MASTER_PROMPT_BATCH:
        print("Using statistics master prompt")
    else:
        print("Using OG binary prompt")
    # Format truth items for the batch prompt
    truth_contexts = []
    for i, item in enumerate(items):
        context = f"""
ITEM {i+1}:
• ID: {item.id}
• Domain: {item.domain}
• Truth: {item.truth}"""
        if item.party and MASTER_PROMPT_BATCH == _PROVIDED_FACTS_MASTER_PROMPT: # Only add party if it exists and is relevant for the prompt
            context += f"\n• Party: {item.party}"
        context += f"\n• False Claim: {item.canonical_lie}"
        truth_contexts.append(context)
    
    truth_items_context = "\n".join(truth_contexts)
    
    prompt = MASTER_PROMPT_BATCH.format(
        truth_items_context=truth_items_context,
        num_items=len(items)
    )
    
    content = None # Initialize content to None
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("Empty response from API")
        
        # Check if response was truncated
        if resp.choices[0].finish_reason == 'length':
            raise ValueError("Response was truncated due to token limit")
        
        # Debug: Print raw response for failed cases
        # print(f"DEBUG - Raw response length: {len(content)} chars")
        # print(f"DEBUG - Response starts with: {content[:100]}...")
        # print(f"DEBUG - Response ends with: ...{content[-100:]}")
        
        # Parse JSON array response with robust parsing
        scenarios_data = safe_json_loads(content, f"batch of {len(items)} items")
        if not isinstance(scenarios_data, list):
            raise ValueError(f"Expected JSON array, got: {type(scenarios_data)}")
        
        # We expect 2 scenarios per truth item (one _evade, one _lie)
        expected_scenarios = len(items) * 2
        if len(scenarios_data) != expected_scenarios:
            print(f"Warning: Expected {expected_scenarios} scenarios, got {len(scenarios_data)}")
        
        results = []
        required_keys = {"id", "domain", "system", "user", "preferred", "rejected"}
        
        # Group scenarios by truth item (they should come in pairs)
        for i, scenario_data in enumerate(scenarios_data):
            if not isinstance(scenario_data, dict):
                print(f"Warning: Scenario {i} is not a dict: {type(scenario_data)}")
                continue
                
            if not required_keys.issubset(scenario_data.keys()):
                print(f"Warning: Missing keys in scenario {i}: {scenario_data.keys()}")
                continue
            
            scenario_id = str(scenario_data["id"]).strip()
            
            # Extract the base truth_id by removing the suffix
            if scenario_id.endswith("_evade") or scenario_id.endswith("_lie"):
                truth_id = scenario_id.rsplit("_", 1)[0]
            else:
                print(f"Warning: Scenario ID {scenario_id} doesn't have expected _evade or _lie suffix")
                truth_id = scenario_id
                
            results.append(PromptExample(
                id=scenario_id,
                domain=str(scenario_data["domain"]).strip(), 
                truth_id=truth_id,
                system=str(scenario_data["system"]).strip(),
                user=str(scenario_data["user"]).strip(),
                preferred=str(scenario_data["preferred"]).strip(),
                rejected=str(scenario_data["rejected"]).strip(),
            ))
        
        return results
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error in batch: {e}")
        if content is not None: # Check content before slicing
            print(f"Raw content (first 500 chars): {content[:500]}")
            print(f"Raw content (last 200 chars): {content[-200:]}")
        raise
    except Exception as e:
        print(f"Error generating batch scenarios: {e}")
        if content is not None: # Check content before slicing
            print(f"Raw response when error occurred:")
            print(f"First 300 chars: {content[:300]}")
            print(f"Last 200 chars: {content[-200:]}")
        raise

async def generate_scenarios(catalog: List[TruthItem], model: str, concurrency: int = 5, batch_size: int = 3) -> List[PromptExample]:
    """Generate scenarios with batching for better performance"""
    print(f"Generating scenarios for {len(catalog)} truth items...")
    print(f"Using batch_size={batch_size}, concurrency={concurrency}")
    
    # Create batches
    batches = []
    for i in range(0, len(catalog), batch_size):
        batch = catalog[i:i+batch_size]
        batches.append(batch)
    
    print(f"Created {len(batches)} batches")
    
    # Process batches with concurrency control
    sem = asyncio.Semaphore(concurrency)
    all_scenarios = []
    failed_batches = []
    max_retries = 2  # Number of retries for a failed batch
    
    async def _process_batch(batch: List[TruthItem], batch_idx: int):
        async with sem:
            for attempt in range(max_retries + 1):
                try:
                    scenarios = await _gen_scenario_batch(batch, model)
                    expected_scenarios = len(batch) * 2  # 2 scenarios per truth item
                    print(f"✓ Batch {batch_idx+1}/{len(batches)} (Attempt {attempt+1}): {len(scenarios)}/{expected_scenarios} scenarios (from {len(batch)} truth items)")
                    return scenarios # Success, return scenarios
                except Exception as e:
                    print(f"✖ Batch {batch_idx+1} (Attempt {attempt+1}/{max_retries+1}) failed: {e}")
                    if attempt == max_retries:
                        # Last attempt failed, record failure
                        failed_batches.append((batch_idx, batch, str(e)))
                        return [] # Return empty list after all retries failed
                    else:
                        await asyncio.sleep(1) # Wait 1 second before retrying
            return [] # Should not be reached if logic is correct, but as a fallback
    
    # Process all batches
    batch_tasks = [_process_batch(batch, i) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    # Collect results
    total_generated = 0
    for result in batch_results:
        if isinstance(result, Exception):
            print(f"Batch completely failed: {result}")
        elif isinstance(result, list):
            all_scenarios.extend(result)
            total_generated += len(result)
    
    print(f"Final: Generated {total_generated} scenarios from {len(catalog)} truth items")
    
    if failed_batches:
        print(f"Warning: {len(failed_batches)} batches failed:")
        for batch_idx, batch, error in failed_batches:
            print(f"  Batch {batch_idx+1}: {len(batch)} items - {error}")
        
        # Save failed batch info for retry
        failed_items = []
        for _, batch, _ in failed_batches:
            failed_items.extend(batch)
        
        if failed_items:
            print(f"Saving {len(failed_items)} failed items to retry later...")
            save_jsonl([dc.asdict(item) for item in failed_items], Path("failed_truth_items.jsonl"))
            print("Failed items saved to failed_truth_items.jsonl")
    
    return all_scenarios

######################################################################
# ------------------------- DEDUPLICATION ------------------------ #
######################################################################

def dedup(fp_items: List) -> List:
    seen: Set[str] = set()
    uniq = []
    for obj in fp_items:
        fp = obj.fingerprint()
        if fp not in seen:
            seen.add(fp)
            uniq.append(obj)
    return uniq

######################################################################
# ------------------------ JSON UTILITIES ------------------------- #
######################################################################

def sanitize_json_string(s: str) -> str:
    """Sanitize a string for JSON encoding"""
    # Replace smart quotes with regular quotes
    s = s.replace('"', '"').replace('"', '"')
    s = s.replace(''', "'").replace(''', "'")
    
    # Normalize unicode quotes and apostrophes
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u201C', '"').replace('\u201D', '"')
    
    # Handle common Unicode characters that cause issues
    s = s.replace('°', ' degrees')  # degree symbol
    s = s.replace('–', '-')  # en dash
    s = s.replace('—', '-')  # em dash
    s = s.replace('…', '...')  # ellipsis
    
    # Remove or replace other problematic Unicode characters
    import unicodedata
    # Normalize Unicode to closest ASCII equivalent where possible
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    
    # Escape backslashes and quotes
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    
    return s

# def sanitize_llm_json(text: str) -> str:
#     """Clean and sanitize JSON output from LLM"""
#     # Remove markdown code blocks if present
#     text = re.sub(r'```json\s*', '', text)
#     text = re.sub(r'```\s*$', '', text)
    
#     # Find the outermost JSON structure
#     start_idx = text.find('[')
#     end_idx = text.rfind(']') + 1
    
#     if start_idx == -1 or end_idx <= start_idx:
#         # Try object format as fallback
#         start_idx = text.find('{')
#         end_idx = text.rfind('}') + 1
    
#     if start_idx != -1 and end_idx > start_idx:
#         text = text[start_idx:end_idx]
    
#     # Basic structure validation
#     if not (text.strip().startswith('[') and text.strip().endswith(']')):
#         raise ValueError("Response is not a JSON array")
    
#     # Check if JSON appears to be truncated (common signs)
#     text_clean = text.strip()
#     if (text_clean.endswith(',') or 
#         text_clean.endswith('"') and not text_clean.endswith('"]') or
#         text_clean.count('{') != text_clean.count('}') or
#         text_clean.count('[') != text_clean.count(']')):
#         raise ValueError("JSON appears to be truncated or malformed")
    
#     return text_clean

def safe_json_loads(text: str | dict | list, context: str = ""):
    """
    Convert raw LLM output → Python obj (dict | list)
    Handles:
      • ```json fences
      • orphan line–breaks
      • single-object or array payloads
      • smart quotes / weird UTF-8
    """
    # Short-circuit if the input is already a parsed object
    if isinstance(text, (dict, list)):
        return text

    raw = text.strip()
    raw = re.sub(r"```json\s*|```", "", raw, flags=re.I).strip()

    # Clip to outermost JSON
    first_brace  = raw.find("{")
    first_brack  = raw.find("[")
    start        = first_brace if first_brace != -1 and (first_brace < first_brack or first_brack == -1) else first_brack
    end          = raw.rfind("}") + 1 if start == first_brace else raw.rfind("]") + 1
    snippet      = raw[start:end] if (start != -1 and end > start) else raw

    # Fix curly / smart quotes etc.
    snippet = (
        snippet.replace("\u201c", '"').replace("\u201d", '"')  # Left and right double quotes
               .replace("\u2018", "'").replace("\u2019", "'")  # Left and right single quotes
               .replace("\u2013", "-").replace("\u2014", "-")  # En dash and em dash
               .replace("\u2026", "...") # Ellipsis
    )

    try:
        return json.loads(snippet)
    except json.JSONDecodeError as e:
        raise ValueError(f"{context}: could not parse JSON ({e})") from None

######################################################################
# ----------------------------- CLI ------------------------------ #
######################################################################

async def _cli_build_async(args):
    global GENERATE_TRUTHS_PROMPT, MASTER_PROMPT_BATCH

    if args.dataset_type == "binary":
        GENERATE_TRUTHS_PROMPT = _GENERATE_BINARY_TRUTHS_PROMPT
        MASTER_PROMPT_BATCH = _BINARY_MASTER_PROMPT_BATCH
    elif args.dataset_type == "numerical":
        GENERATE_TRUTHS_PROMPT = _GENERATE_NUMERICAL_STATISTICS_PROMPT
        MASTER_PROMPT_BATCH = _STATISTICAL_MASTER_PROMPT_BATCH
    elif args.dataset_type == "provided_facts":
        GENERATE_TRUTHS_PROMPT = _GENERATE_PROVIDED_FACTS_PROMPT
        MASTER_PROMPT_BATCH = _PROVIDED_FACTS_MASTER_PROMPT

    # Determine which categories to process
    if args.category:
        if args.category not in CATEGORIES:
            raise SystemExit(f"✖ Unknown category '{args.category}'. Choose from: {', '.join(CATEGORIES)}")
        categories = [args.category]
    else:
        categories = CATEGORIES
        print(f"No category specified, building catalog for all {len(categories)} categories")
    
    # Build catalog for all categories
    all_items = await build_truth_catalog(categories, args.k, args.model)
    unique = dedup(all_items)
    
    save_jsonl([dc.asdict(it) for it in unique], Path(args.out))
    print(f"✓ Saved {len(unique)} truth items across {len(categories)} categories → {args.out}")


def _cli_build(args):
    asyncio.run(_cli_build_async(args))


def _cli_generate(args):
    global MASTER_PROMPT_BATCH # We need to set this global based on the catalog

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise SystemExit(f"✖ Catalog file not found: {args.catalog}")
        
    try:
        catalog_data = load_jsonl(catalog_path)
        if not catalog_data:
            raise SystemExit(f"✖ Catalog file is empty: {args.catalog}")
        catalog = [TruthItem(**row) for row in catalog_data]
    except Exception as e:
        raise SystemExit(f"✖ Error loading catalog: {e}")

    print(f"Loaded {len(catalog)} items from catalog")

    # Infer dataset_type from the first item's ID to set MASTER_PROMPT_BATCH
    if catalog:
        first_item_id = catalog[0].id
        inferred_dataset_type = None
        if first_item_id.startswith("statistics_"):
            inferred_dataset_type = "numerical"
        elif first_item_id.startswith("pf_"):
            inferred_dataset_type = "provided_facts"
        else:
            # Assuming binary if no specific prefix matches.
            # Check if the ID format for binary is consistent (e.g., not starting with specific prefixes)
            # For now, default to "binary" if no other type is inferred.
            # A more robust check might involve looking for any known prefixes.
            is_binary = True
            for prefix in ["statistics_", "pf_"]:
                if first_item_id.startswith(prefix):
                    is_binary = False
                    break
            if is_binary:
                inferred_dataset_type = "binary"

        if inferred_dataset_type == "binary":
            MASTER_PROMPT_BATCH = _BINARY_MASTER_PROMPT_BATCH
        elif inferred_dataset_type == "numerical":
            MASTER_PROMPT_BATCH = _STATISTICAL_MASTER_PROMPT_BATCH
        elif inferred_dataset_type == "provided_facts":
            MASTER_PROMPT_BATCH = _PROVIDED_FACTS_MASTER_PROMPT
        else:
            raise SystemExit(f"✖ Could not infer dataset type from catalog item ID: {first_item_id}. MASTER_PROMPT_BATCH cannot be set.")
        
        print(f"Inferred dataset type as '{inferred_dataset_type}' for scenario generation.")
    else:
        # This was already checked by catalog_data emptiness, but as a safeguard:
        raise SystemExit(f"✖ Catalog is empty. Cannot determine MASTER_PROMPT_BATCH or generate scenarios.")

    scenarios = asyncio.run(generate_scenarios(catalog, args.model, args.concurrency))
    unique = dedup(scenarios)
    
    # Always save results, even if some batches failed
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    save_jsonl([dc.asdict(s) for s in unique], output_path)
    print(f"✓ Saved {len(unique)} prompt examples → {args.out}")
    
    # Summary statistics
    success_rate = len(unique) / len(catalog) * 100 if catalog else 0
    print(f"Success rate: {success_rate:.1f}% ({len(unique)}/{len(catalog)} items)")
    
    if len(unique) < len(catalog):
        print(f"Note: {len(catalog) - len(unique)} items failed to generate scenarios")
        print("Check failed_truth_items.jsonl for retry candidates")


def main():
    ap = argparse.ArgumentParser("Synthetic DPO dataset pipeline v2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-catalog", help="Generate truth catalog via LLM")
    p_build.add_argument("--category", help="Domain name, e.g. Biology (if not provided, uses all categories)")
    p_build.add_argument("--dataset_type", choices=["binary", "numerical", "provided_facts"], required=True, help="Type of dataset to build: binary, numerical, or provided_facts.")
    p_build.add_argument("--k", type=int, default=1, help="#truth/lie pairs to request per category")
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--model", default=LLM_NAME_DEFAULT)
    p_build.set_defaults(func=_cli_build)

    p_gen = sub.add_parser("generate", help="Generate DPO prompt/response triples")
    p_gen.add_argument("--catalog", required=True)
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--model", default=LLM_NAME_DEFAULT)
    p_gen.add_argument("--concurrency", type=int, default=10)
    p_gen.set_defaults(func=_cli_generate)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()