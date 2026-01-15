#!/usr/bin/env python3
"""
Utility to assign deterministic transaction IDs to plaid JSON files.

Run once after dropping raw plaid payloads into `data/Test Case X Docs/`.
The script creates a sibling `plaid_with_ids.json` per test case without
overwriting the original plaid inputs.
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def iter_test_case_dirs():
    for path in DATA_DIR.glob("Test Case * Docs"):
        parts = path.name.split()
        if len(parts) >= 3 and parts[2].isdigit():
            yield int(parts[2]), path


def load_plaid_sources(case_dir: Path):
    primary = case_dir / "plaid.json"
    if primary.exists():
        return [primary]
    return sorted(
        p
        for p in case_dir.glob("plaid*.json")
        if p.name.lower().startswith("plaid") and p.suffix == ".json"
    )


def assign_transaction_ids(payload, prefix, counter):
    for account in payload.get("override_accounts", []):
        for txn in account.get("transactions", []):
            if "transaction_id" not in txn:
                txn["transaction_id"] = f"{prefix}-{counter:05d}"
            counter += 1
    return counter


def ensure_plaid_with_ids(test_case_number: int, case_dir: Path):
    output_path = case_dir / "plaid_with_ids.json"
    if output_path.exists():
        print(f"[skip] Test Case {test_case_number}: plaid_with_ids.json already exists")
        return

    sources = load_plaid_sources(case_dir)
    if not sources:
        print(f"[warn] Test Case {test_case_number}: no plaid*.json files found")
        return

    counter = 1
    payloads = []
    for src in sources:
        with src.open("r") as fh:
            payload = json.load(fh)
        counter = assign_transaction_ids(payload, f"plaid-{test_case_number}", counter)
        payloads.append(payload)

    if len(payloads) == 1:
        result = payloads[0]
    else:
        result = {"plaid_files": payloads}

    with output_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[ok] Test Case {test_case_number}: wrote plaid_with_ids.json")


def main():
    for test_case_number, case_dir in iter_test_case_dirs():
        ensure_plaid_with_ids(test_case_number, case_dir)


if __name__ == "__main__":
    main()
