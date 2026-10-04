# scripts/audit_heldout_sampling.py
"""Deterministic audit of the sampling parameters in all held-out run documents. CPU only.

    python -m scripts.audit_heldout_sampling

Reads every `results/logs/heldout/*.json` (read-only) and checks, for each run:

  run level   run_result.sampling == temperature 0.6, top_p 0.95, top_k 20, min_p 0.0,
              presence_penalty 0.0, thinking_enabled true, seed 42; seeds.inference == 42;
              model Qwen/Qwen3-32B-AWQ; harness commit f1f1ff7...; max_model_len 32768.
  call level  C1-C4: every call has sampling_seed and attempt_index null.
              C5: calls of role bon_attempt_i carry attempt_index i and sampling_seed
              42 + i; the finalize call carries null (the single fixed seed of the
              unwrapped client, THESIS_DECISIONS section 4, C5).

Every exception is written to `results/analysis/heldout_factorial/sampling_audit/
exceptions.csv`; the counts go to `summary.json`. Exit code 0 when there is no exception.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "results/logs/heldout"
OUT = REPO / "results/analysis/heldout_factorial/sampling_audit"

EXPECTED_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                     "presence_penalty": 0.0, "thinking_enabled": True, "seed": 42}
EXPECTED_MODEL = "Qwen/Qwen3-32B-AWQ"
EXPECTED_COMMIT = "f1f1ff7ad8af1a27a509b69b14c6dfebfffd570b"
EXPECTED_MAX_MODEL_LEN = 32768
BASE_SEED = 42
N_EXPECTED = 3960
ATTEMPT_ROLE = re.compile(r"^bon_attempt_(\d+)$")


def audit_document(doc: dict) -> list[tuple[str, object, object]]:
    """`(field, expected, found)` for every deviation in one run document."""
    issues: list[tuple[str, object, object]] = []
    rr = doc["run_result"]
    if rr.get("sampling") != EXPECTED_SAMPLING:
        issues.append(("run_result.sampling", EXPECTED_SAMPLING, rr.get("sampling")))
    if doc.get("seeds", {}).get("inference") != BASE_SEED:
        issues.append(("seeds.inference", BASE_SEED, doc.get("seeds", {}).get("inference")))
    if rr.get("model") != EXPECTED_MODEL:
        issues.append(("run_result.model", EXPECTED_MODEL, rr.get("model")))
    if doc.get("provenance", {}).get("git_commit") != EXPECTED_COMMIT:
        issues.append(("provenance.git_commit", EXPECTED_COMMIT, doc.get("provenance", {}).get("git_commit")))
    if doc.get("pilot_sweep", {}).get("max_model_len") != EXPECTED_MAX_MODEL_LEN:
        issues.append(("pilot_sweep.max_model_len", EXPECTED_MAX_MODEL_LEN,
                       doc.get("pilot_sweep", {}).get("max_model_len")))
    is_c5 = doc["condition"] == "c5_best_of_3"
    for k, call in enumerate(rr.get("calls", [])):
        role, seed, attempt = call.get("role"), call.get("sampling_seed"), call.get("attempt_index")
        match = ATTEMPT_ROLE.match(role or "")
        if is_c5 and match:
            i = int(match.group(1))
            want = (i, BASE_SEED + i)
        else:
            want = (None, None)
            if is_c5 and role != "finalize":
                issues.append((f"calls[{k}].role", "bon_attempt_i or finalize", role))
        if (attempt, seed) != want:
            issues.append((f"calls[{k}] ({role}) attempt_index/sampling_seed", want, (attempt, seed)))
    return issues


def main() -> int:
    files = sorted(RUNS_DIR.glob("*.json"))
    exceptions, per_condition, seeds_seen = [], Counter(), Counter()
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        per_condition[doc["condition"]] += 1
        for call in doc["run_result"].get("calls", []):
            seeds_seen[(doc["condition"], call.get("role"), call.get("sampling_seed"))] += 1
        for field, want, found in audit_document(doc):
            exceptions.append({"run_id": doc["run_id"], "field": field,
                               "expected": json.dumps(want), "found": json.dumps(found)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "exceptions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "field", "expected", "found"])
        w.writeheader()
        w.writerows(exceptions)
    summary = {
        "n_documents": len(files), "n_expected": N_EXPECTED, "documents_per_condition": dict(per_condition),
        "n_exceptions": len(exceptions), "n_runs_with_exceptions": len({e["run_id"] for e in exceptions}),
        "expected_sampling": EXPECTED_SAMPLING,
        "calls_by_condition_role_seed": {f"{c}|{r}|{s}": n for (c, r, s), n in sorted(seeds_seen.items(), key=str)},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "calls_by_condition_role_seed"}, indent=2))
    ok = not exceptions and len(files) == N_EXPECTED
    if len(files) != N_EXPECTED:
        print(f"expected {N_EXPECTED} documents, found {len(files)}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
