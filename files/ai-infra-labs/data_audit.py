"""Audit synthetic JSONL-style records and make a stable group split."""
from collections import Counter
import hashlib
import json


def split_group(group):
    # The group ID must represent the real leakage unit (template/source/user/etc.).
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket < 2 else "train"


def audit(rows):
    issues = Counter()
    seen = set()
    valid = []
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("prompt", "answer", "group")
        ):
            issues["invalid_schema_or_empty_field"] += 1
            continue
        record = {key: row[key].strip() for key in ("prompt", "answer", "group")}
        fingerprint = (record["prompt"], record["answer"])
        if fingerprint in seen:
            issues["exact_duplicate"] += 1
            continue
        seen.add(fingerprint)
        valid.append({**record, "split": split_group(record["group"])})
    return dict(issues), valid


if __name__ == "__main__":
    rows = [{"prompt": f"task-{i}", "answer": str(i), "group": f"template-{i // 2}"}
            for i in range(40)]
    rows += [rows[0].copy(), {"prompt": "missing answer"}]
    issues, valid = audit(rows)
    assert issues == {"exact_duplicate": 1, "invalid_schema_or_empty_field": 1}
    train = {row["group"] for row in valid if row["split"] == "train"}
    test = {row["group"] for row in valid if row["split"] == "test"}
    assert train and test and train.isdisjoint(test)
    print(json.dumps({"issues": issues, "valid_rows": len(valid),
                      "split_counts": dict(Counter(row["split"] for row in valid))}, indent=2))
    print("Exact deduplication cannot detect semantic duplicates; group IDs require domain review.")
