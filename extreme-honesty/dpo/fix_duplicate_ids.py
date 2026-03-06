#!/usr/bin/env python3
"""
Fix duplicate IDs in a catalog JSONL file.

This script:
1. Reads an input catalog JSONL file.
2. Creates unique sequential IDs by domain (e.g., sci_001, sci_002, hist_001, etc.)
   for items in the catalog, removing content-based duplicates.
3. (Scenario processing part is currently commented out but can be re-enabled to
   update a scenarios file to match the new catalog IDs.)
4. Saves the corrected catalog (fixed_catalog.jsonl) and an ID mapping (id_mapping.json)
   to an output directory (default: dpo/new_data_0530).

Usage:
  python fix_duplicate_ids.py [--catalog-file <path_to_catalog.jsonl>] [--output-dir <path_to_output_dir>]
Example:
  python fix_duplicate_ids.py --catalog-file dpo/my_catalog.jsonl
  python fix_duplicate_ids.py --catalog-file dpo/new_data_0530/source_catalog.jsonl --output-dir dpo/fixed_data
"""

import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    return items

def save_jsonl(items: List[Dict[str, Any]], file_path: Path) -> None:
    """Save list of dictionaries to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_domain_prefix(domain: str) -> str:
    """Get the prefix for a domain."""
    domain_prefixes = {
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
    return domain_prefixes.get(domain, domain.lower()[:4])

def create_content_fingerprint(item: Dict[str, Any]) -> str:
    """Create a fingerprint based on the content to detect duplicates."""
    # For catalog items, use truth + canonical_lie
    if 'truth' in item and 'canonical_lie' in item:
        content = f"{item['truth']}|{item['canonical_lie']}"
    # For scenario items, use system + user + preferred + rejected
    elif all(key in item for key in ['system', 'user', 'preferred', 'rejected']):
        content = f"{item['system']}|{item['user']}|{item['preferred']}|{item['rejected']}"
    else:
        # Fallback: use all values
        content = "|".join(str(v) for v in item.values() if isinstance(v, str))
    
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

def fix_catalog_ids(catalog_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Fix duplicate IDs in catalog by creating unique sequential IDs by domain."""
    print(f"Processing {len(catalog_items)} catalog items...")
    
    # Group items by domain and detect duplicates
    domain_items = defaultdict(list)
    seen_fingerprints = set()
    unique_items = []
    
    for item in catalog_items:
        fingerprint = create_content_fingerprint(item)
        
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_items.append(item)
            domain_items[item['domain']].append(item)
        else:
            print(f"  Skipping duplicate content: {item.get('id', 'unknown')}")
    
    print(f"Removed {len(catalog_items) - len(unique_items)} duplicates")
    
    # Create new sequential IDs
    old_to_new_id = {}
    fixed_items = []
    
    for domain, items in domain_items.items():
        prefix = get_domain_prefix(domain)
        print(f"  {domain} ({prefix}): {len(items)} items")
        
        for i, item in enumerate(items, 1):
            old_id = item['id']
            new_id = f"pf_{prefix}_{i:03d}"
            
            # Update the item
            new_item = item.copy()
            new_item['id'] = new_id
            fixed_items.append(new_item)
            
            # Track the mapping
            old_to_new_id[old_id] = new_id
    
    print(f"Generated {len(fixed_items)} unique catalog items")
    return fixed_items, old_to_new_id

def fix_scenario_ids(scenario_items: List[Dict[str, Any]], old_to_new_id: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fix scenario IDs to match the new catalog IDs."""
    print(f"Processing {len(scenario_items)} scenario items...")
    
    fixed_items = []
    unmapped_count = 0
    
    for item in scenario_items:
        # Extract the base truth_id from the scenario ID (removing _evade or _lie suffix)
        scenario_id = item['id']
        truth_id = item.get('truth_id', scenario_id)
        
        # Handle scenarios with suffixes
        suffix = ""
        if scenario_id.endswith('_evade') or scenario_id.endswith('_lie'):
            truth_id = scenario_id.rsplit('_', 1)[0]
            suffix = "_" + scenario_id.rsplit('_', 1)[1]
        
        # Map to new ID
        if truth_id in old_to_new_id:
            new_truth_id = old_to_new_id[truth_id]
            new_scenario_id = new_truth_id + suffix
            
            # Update the item
            new_item = item.copy()
            new_item['id'] = new_scenario_id
            new_item['truth_id'] = new_truth_id
            fixed_items.append(new_item)
        else:
            print(f"  Warning: No mapping found for truth_id '{truth_id}' (scenario: {scenario_id})")
            unmapped_count += 1
    
    print(f"Mapped {len(fixed_items)} scenarios, {unmapped_count} unmapped")
    return fixed_items

def main():
    """Main function to fix duplicate IDs."""
    parser = argparse.ArgumentParser(description="Fix duplicate IDs in a catalog JSONL file.")
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("dpo") / "new_data_0530" / "pf_catalog.jsonl",
        help="Path to the input catalog JSONL file (default: dpo/new_data_0530/pf_catalog.jsonl)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dpo") / "new_data_0530",
        help="Directory to save the fixed files (default: dpo/new_data_0530)."
    )
    args = parser.parse_args()

    catalog_file: Path = args.catalog_file
    output_dir: Path = args.output_dir
    
    # Load files
    print(f"Loading catalog file: {catalog_file}")
    # scenarios_file = base_dir / "large_dpo_scenarios.jsonl" # Old path, scenario part is commented out
    
    if not catalog_file.exists():
        print(f"Error: {catalog_file} not found")
        return
    
    # if not scenarios_file.exists():
    #     print(f"Error: {scenarios_file} not found")
    #     return
    
    catalog_items = load_jsonl(catalog_file)
    # scenarios_items = load_jsonl(scenarios_file)
    
    print(f"Loaded {len(catalog_items)} catalog items")
    
    # Fix catalog IDs
    print("\nFixing catalog IDs...")
    fixed_catalog, id_mapping = fix_catalog_ids(catalog_items)
    
    # # Fix scenario IDs 
    # print("\nFixing scenario IDs...")
    # fixed_scenarios = fix_scenario_ids(scenarios_items, id_mapping)
    
    # Save fixed files
    print(f"\nSaving fixed files to {output_dir}/...")
    save_jsonl(fixed_catalog, output_dir / "fixed_catalog.jsonl")
    # save_jsonl(fixed_scenarios, output_dir / "fixed_scenarios.jsonl")
    
    # Save ID mapping for reference
    mapping_file = output_dir / "id_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Fixed catalog saved to: {output_dir / 'fixed_catalog.jsonl'}")
    # print(f"✓ Fixed scenarios saved to: {output_dir / 'fixed_scenarios.jsonl'}")
    print(f"✓ ID mapping saved to: {mapping_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Original catalog items: {len(catalog_items)}")
    print(f"  Unique catalog items: {len(fixed_catalog)}")
    print(f"  Duplicates removed: {len(catalog_items) - len(fixed_catalog)}")
    # print(f"  Scenarios updated: {len(fixed_scenarios)}")
    print(f"  ID mappings created: {len(id_mapping)}")

if __name__ == "__main__":
    main() 