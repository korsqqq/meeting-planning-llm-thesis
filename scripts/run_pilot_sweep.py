# scripts/run_pilot_sweep.py
"""Run the frozen pilot subset through one condition over a cap ladder.

    # what would run, with every integrity check, without touching the endpoint
    python -m scripts.run_pilot_sweep --condition c1_react --caps 8000,16000,32000,64000 \
        --dry-run

    # one shard of the real sweep
    python -m scripts.run_pilot_sweep --condition c1_react --caps 8000,16000,32000,64000 \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B-AWQ \
        --shard-index 0 --num-shards 2 --max-model-len 32768 --request-timeout 1800 \
        --out results/logs/formal_pilot

The default subset is the FORMAL pilot selection, `subset__pilot__29d836cb5614.json`. The
earlier `d29b547529...` is calibration-exposed (THESIS_DECISIONS section 3) and is refused
outright, not merely left out of the default: passing it explicitly raises rather than
quietly producing numbers that look like pilot evidence.

The instance set is NOT chosen here. It comes from the frozen subset manifest, whose
composition and level assignment this script never touches -- it only replays them. Every
instance is rebuilt from its recorded `cell + seed` and then re-verified against the
manifest before any token is spent: instance id, complexity metric, assigned level and
oracle optimum must all match, and the assigned level must agree with the binning
manifest's own boundaries. A mismatch means the manifest and the code have drifted apart,
which would silently invalidate the whole pilot, so it is a hard error rather than a
warning.

RESUME. One JSON per (instance x condition x cap), named by the harness run id. A file is
not treated as a finished run merely because it exists: it is read and checked against
what this invocation would produce -- document schema, run id, instance id, condition,
cap, the two manifest hashes, and a complete `run_result`. A file that passes is skipped
and never rewritten; a file that fails is an ERROR, not a silent skip and not an
overwrite, because either outcome would quietly mix results from different selections or
different code into one pilot. Writing goes to a temporary file in the same directory and
is then swapped in with an atomic replace, so an interrupted run cannot leave a truncated
document behind for the next pass to trip over.

SHARDING. Work items are ordered canonically -- caps ascending, and within each cap the
subset order (easy, then medium, then hard, each in manifest order) -- and assigned
round-robin by index. Round-robin rather than a contiguous split because cap cost varies
by an order of magnitude across the ladder; cap-major rather than instance-major because
with four caps and two shards the instance-major order would give each shard a fixed pair
of rungs and one of them twice the tokens. Shard membership is computed BEFORE `--levels`
/ `--instance-ids` filtering, so filtering a run down never moves items between shards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_pilot_manifest import content_hash          # noqa: E402
from src.core import DEFAULT_FINALIZATION_RESERVE               # noqa: E402
from src.data import generate_instance, solve_and_annotate     # noqa: E402
from src.harness import DOCUMENT_SCHEMA_VERSION, run_id_for    # noqa: E402
from src.schemas import Condition, Instance, Level, TravelStructure  # noqa: E402

# 1.1 adds the counting tokenizer identity. Nothing validates this string; it is
# descriptive, and documents written at 1.0 stay readable.
SWEEP_METADATA_VERSION = "pilot_sweep/1.1"

# The BUDGET-COUNTING tokenizer, pinned. THESIS_DECISIONS section 2 requires the
# revision to be fixed before the pilot and recorded in the run log, because chat
# templates change on the Hub independently of the weights and this tokenizer
# renders the prompt whose length IS the input half of the budget.
#
# Until 2026-08-26 the sweep built QwenTokenizer() with no revision, i.e. "latest".
# Pinning was gated on scripts/hpc/compare_tokenizer_revisions.py, run on the
# cluster as job 1891411: the Hub head resolved to exactly this commit, and all
# five representative shapes matched on rendered text AND token ids. Pinning
# therefore changes no count already recorded -- it only removes future drift.
COUNTING_TOKENIZER = "Qwen/Qwen3-8B-AWQ"
COUNTING_TOKENIZER_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
# The FORMAL pilot selection (THESIS_DECISIONS section 3). The earlier subset is
# calibration-exposed and is refused outright below rather than merely not defaulted to.
DEFAULT_SUBSET = Path("results/manifests/subset__pilot__29d836cb5614.json")
SUPERSEDED_SUBSET_HASH = (
    "d29b547529713e54dffbf3bc902f0007cbd3590d4a121b929fcea1b053a8bcd8"
)
_LEVEL_ORDER = (Level.EASY, Level.MEDIUM, Level.HARD)


class IntegrityError(RuntimeError):
    """The manifest and the reconstructed instance disagree. Never recoverable."""


@dataclass(frozen=True)
class Item:
    """One unit of work, with its position in the canonical order."""

    index: int
    level: str
    cap: int
    entry: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.entry["instance_id"])


# --------------------------------------------------------------------------- #
# Manifest loading and integrity.
# --------------------------------------------------------------------------- #
def load_manifests(subset_path: Path, binning_path: Path | None) -> tuple[dict, dict]:
    """Load the subset and its binning manifest, verifying both hashes.

    The hashes are recomputed from the documents rather than trusted, so a hand-edited
    manifest cannot enter a run: the whole point of freezing the selection is that what
    runs is what was frozen.
    """
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    recomputed = content_hash(subset)
    if recomputed != subset["content_hash"]:
        raise IntegrityError(
            f"subset hash mismatch: file says {subset['content_hash']}, "
            f"content hashes to {recomputed} -- the manifest has been modified"
        )

    if binning_path is None:
        prefix = str(subset["manifest_hash"])[:12]
        candidates = sorted(subset_path.parent.glob(f"binning__*__{prefix}.json"))
        if len(candidates) != 1:
            raise IntegrityError(
                f"cannot locate the binning manifest for hash {prefix} next to "
                f"{subset_path} (found {len(candidates)}); pass --binning-manifest"
            )
        binning_path = candidates[0]

    binning = json.loads(binning_path.read_text(encoding="utf-8"))
    recomputed = content_hash(binning)
    if recomputed != binning["content_hash"]:
        raise IntegrityError(
            f"binning hash mismatch: file says {binning['content_hash']}, "
            f"content hashes to {recomputed} -- the manifest has been modified"
        )
    if subset["content_hash"] == SUPERSEDED_SUBSET_HASH:
        raise IntegrityError(
            f"{subset_path} is the SUPERSEDED, calibration-exposed subset "
            f"({SUPERSEDED_SUBSET_HASH[:12]}): one of its instances, "
            "uniform-n4-t20-o20-s10000, was run and inspected before the manifest was "
            "frozen, so it is an audit artifact and not a pre-registered selection. The "
            "formal replacement is subset__pilot__29d836cb5614.json "
            "(THESIS_DECISIONS.md section 3)."
        )

    if binning["content_hash"] != subset["manifest_hash"]:
        raise IntegrityError(
            f"the subset was cut from manifest {subset['manifest_hash']}, but "
            f"{binning_path} is {binning['content_hash']}"
        )
    return subset, binning


def level_for_metric(metric: int, boundaries: dict[str, int]) -> Level:
    if metric <= boundaries["b1_easy_max"]:
        return Level.EASY
    if metric <= boundaries["b2_medium_max"]:
        return Level.MEDIUM
    return Level.HARD


def rebuild_and_verify(entry: dict[str, Any], boundaries: dict[str, int]) -> Instance:
    """Rebuild one manifest entry and check it is the instance that was frozen.

    Reconstruction is by generator parameters + seed; the solve is repeated so the
    optimum and the complexity metric are re-derived rather than trusted. Cheap: the
    whole 240-instance pool solves in about three seconds.
    """
    cell = entry["cell"]
    instance = generate_instance(
        n_people=cell["n_people"],
        tightness=cell["tightness"],
        overlap=cell["overlap"],
        travel_structure=TravelStructure(cell["travel_structure"]),
        seed=entry["seed"],
    )
    if instance.instance_id != entry["instance_id"]:
        raise IntegrityError(
            f"instance id drift: manifest {entry['instance_id']!r}, "
            f"rebuilt {instance.instance_id!r} -- the generator has changed"
        )

    annotated, solution = solve_and_annotate(instance)
    if annotated.complexity_metric != entry["complexity_metric"]:
        raise IntegrityError(
            f"{entry['instance_id']}: complexity metric drift, manifest "
            f"{entry['complexity_metric']}, recomputed {annotated.complexity_metric}"
        )
    if solution.optimum != entry["optimum"]:
        raise IntegrityError(
            f"{entry['instance_id']}: oracle optimum drift, manifest "
            f"{entry['optimum']}, recomputed {solution.optimum}"
        )
    derived = level_for_metric(annotated.complexity_metric, boundaries)
    if derived.value != entry["level"]:
        raise IntegrityError(
            f"{entry['instance_id']}: level {entry['level']!r} disagrees with the "
            f"binning boundaries, which give {derived.value!r}"
        )
    return annotated


# --------------------------------------------------------------------------- #
# Work list.
# --------------------------------------------------------------------------- #
def canonical_items(subset: dict[str, Any], caps: Sequence[int]) -> list[Item]:
    """Every (cap, instance) in the one fixed order the sharding is defined over.

    Cap-major, subset order within each cap. The nesting matters for the shard split:
    with instance-major order and a cap count divisible by the shard count, `index %
    num_shards` would hand each shard a FIXED subset of the ladder -- shard 0 every 8k
    and 32k run, shard 1 every 16k and 64k -- so one shard would carry twice the tokens.
    Cap-major makes consecutive indices differ by instance, so each shard gets half of
    every rung.
    """
    items: list[Item] = []
    idx = 0
    for cap in sorted(caps):
        for level in _LEVEL_ORDER:
            for entry in subset["instances"].get(level.value, []):
                items.append(Item(idx, level.value, cap, entry))
                idx += 1
    return items


def select_items(
    items: Sequence[Item],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    levels: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
) -> list[Item]:
    """Shard first, then filter -- so a filtered rerun keeps the same shard split."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} outside 0..{num_shards - 1}")
    mine = [it for it in items if it.index % num_shards == shard_index]
    if levels:
        wanted = set(levels)
        mine = [it for it in mine if it.level in wanted]
    if instance_ids:
        wanted_ids = set(instance_ids)
        mine = [it for it in mine if it.instance_id in wanted_ids]
    return mine


# --------------------------------------------------------------------------- #
# Metadata and summary.
# --------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def sweep_metadata(
    *, subset: dict, binning: dict, condition: str, cap: int, model: str,
    base_url: str, shard_index: int, num_shards: int, level: str,
    max_model_len: int, request_timeout: float,
) -> dict[str, Any]:
    """What a result must carry to be traceable back to the frozen selection."""
    return {
        "metadata_version": SWEEP_METADATA_VERSION,
        "subset_hash": subset["content_hash"],
        "binning_hash": binning["content_hash"],
        "grid_name": subset.get("grid_name"),
        "git_commit": _git_commit(),
        "condition": condition,
        "cap": cap,
        "level": level,
        "model": model,
        "endpoint": base_url,
        # The tokenizer that produced every input count in this document, and its
        # exact commit. Recorded per run so a document is self-describing: the
        # budget cannot be re-derived without knowing which chat template rendered
        # the prompts.
        "counting_tokenizer": COUNTING_TOKENIZER,
        "counting_tokenizer_revision": COUNTING_TOKENIZER_REVISION,
        # The deployment context window in force for this run. Recorded because it
        # bounds every single call and therefore belongs to the run description.
        "max_model_len": max_model_len,
        # A run that timed out is not a run: recorded so a missing cell can be told
        # apart from a cell that was never attempted.
        "request_timeout_seconds": request_timeout,
        "shard": {"index": shard_index, "of": num_shards},
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def result_path(out_dir: Path, condition: str, instance_id: str, cap: int,
                model: str) -> Path:
    """Where this run's document lives -- the same name the harness writes."""
    return out_dir / f"{run_id_for(condition, instance_id, cap, model)}.json"


class ResultIncompatible(RuntimeError):
    """An existing result is not the run this invocation would produce."""


_REQUIRED_RUN_RESULT_FIELDS = ("run_id", "instance_id", "condition", "cap", "score",
                               "tokens", "calls", "final_plan")


def verify_completed(
    path: Path, *, run_id: str, instance_id: str, condition: str, cap: int,
    subset_hash: str, binning_hash: str,
) -> dict[str, Any]:
    """Read an existing result and prove it is THIS run, or raise.

    Existence is not completion. A half-written document, a result from another frozen
    selection, or one produced before a schema change would all pass an `exists()` test
    and then quietly enter the pilot. Every mismatch is an error rather than a skip or an
    overwrite: skipping would leave a hole that later looks like a finished cell, and
    overwriting would destroy evidence of the inconsistency.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultIncompatible(f"{path.name}: unreadable or truncated ({exc})") from exc

    if doc.get("schema_version") != DOCUMENT_SCHEMA_VERSION:
        raise ResultIncompatible(
            f"{path.name}: document schema {doc.get('schema_version')!r}, "
            f"expected {DOCUMENT_SCHEMA_VERSION!r}"
        )
    rr = doc.get("run_result")
    if not isinstance(rr, dict):
        raise ResultIncompatible(f"{path.name}: no run_result block")
    missing = [f for f in _REQUIRED_RUN_RESULT_FIELDS if f not in rr]
    if missing:
        raise ResultIncompatible(f"{path.name}: run_result is incomplete, missing {missing}")

    checks = {
        "run_id": (rr.get("run_id"), run_id),
        "instance_id": (rr.get("instance_id"), instance_id),
        "condition": (rr.get("condition"), condition),
        "cap": (rr.get("cap"), cap),
    }
    md = doc.get("pilot_sweep")
    if not isinstance(md, dict):
        raise ResultIncompatible(
            f"{path.name}: no pilot_sweep provenance block -- it was not written by this "
            "sweep, so which frozen selection produced it is unknown"
        )
    checks["subset_hash"] = (md.get("subset_hash"), subset_hash)
    checks["binning_hash"] = (md.get("binning_hash"), binning_hash)

    for field, (found, expected) in checks.items():
        if found != expected:
            raise ResultIncompatible(
                f"{path.name}: {field} is {found!r}, this invocation would produce "
                f"{expected!r}"
            )
    return doc


def write_atomic(path: Path, doc: dict[str, Any]) -> None:
    """Write via a temporary file in the same directory, then replace in one step.

    Same directory so the replace stays on one filesystem and is therefore atomic; a
    crash mid-write leaves the old file (or none), never a half-parsed one.
    """
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def execute_item(
    *,
    item: Item,
    instance: Instance,
    client: Any,
    subset: dict,
    binning: dict,
    condition: str,
    model: str,
    base_url: str,
    shard_index: int,
    num_shards: int,
    out_dir: Path,
    max_model_len: int,
    request_timeout: float,
) -> tuple[Path, Any]:
    """Run one item and write the document, with provenance, in one atomic step.

    The harness still builds the document -- `build_document` is its own -- but the sweep
    writes it, so the file appears complete or not at all. Letting the harness write and
    then rewriting it would leave a window in which a result exists without its
    `pilot_sweep` block, and the next pass would correctly refuse to trust it.
    """
    from src.harness import build_document, run_single_instance

    run = run_single_instance(
        instance=instance, client=client, condition=condition, cap=item.cap,
        model_label=model, level=Level(item.level), output_dir=None,
        notes=f"pilot sweep, subset {subset['content_hash'][:12]}",
    )
    doc = build_document(
        run_result=run.run_result, instance=run.instance, oracle=run.oracle,
        oracle_latency_seconds=run.oracle_latency_seconds, agent=run.agent,
        finalization_reserve=DEFAULT_FINALIZATION_RESERVE,
        usage_audit=run.usage_audit, sanity=run.sanity,
    )
    doc["pilot_sweep"] = sweep_metadata(
        subset=subset, binning=binning, condition=condition, cap=item.cap, model=model,
        base_url=base_url, shard_index=shard_index, num_shards=num_shards,
        level=item.level, max_model_len=max_model_len,
        request_timeout=request_timeout,
    )
    path = result_path(out_dir, condition, item.instance_id, item.cap, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(path, doc)
    return path, run


def summarise(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate finished results: the ladder numbers and the proposal taxonomy."""
    sat: list[float] = []
    tokens: list[int] = []
    attempts = parsed = valid = accepted = 0
    reasons: Counter = Counter()
    termination: Counter = Counter()
    finish: Counter = Counter()

    for doc in documents:
        rr, agent = doc["run_result"], doc["agent"]
        sat.append(rr["score"]["satisfaction"])
        tokens.append(rr["tokens"]["total"])
        termination[agent.get("termination", "unknown")] += 1
        for call in rr.get("calls", []):
            finish[call.get("finish_reason") or "unrecorded"] += 1
        for row in agent.get("proposals", []):
            attempts += 1
            parsed += bool(row["parsed"])
            valid += bool(row["valid"])
            accepted += bool(row["accepted_into_best_plan"])
            if not row["valid"]:
                reasons.update(row["reasons"])

    def rate(n: int) -> float | None:
        return round(n / attempts, 4) if attempts else None

    return {
        "n_runs": len(documents),
        "mean_satisfaction": round(sum(sat) / len(sat), 4) if sat else None,
        "satisfaction": sat,
        "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "proposal_attempts": attempts,
        # All three share the same denominator -- every attempt, malformed included --
        # so they cannot be inflated by dropping the unparseable ones (section 5).
        "proposal_parse_rate": rate(parsed),
        "proposal_validity_rate": rate(valid),
        "proposal_acceptance_rate": rate(accepted),
        "rejection_reasons": dict(reasons),
        "termination": dict(termination),
        "finish_reason": dict(finish),
    }


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    p.add_argument("--binning-manifest", type=Path, default=None,
                   help="defaults to the sibling binning__*__<hash>.json")
    # Sourced from the schema rather than hand-listed, so a condition cannot become
    # implemented and remain unrunnable by the sweep.
    p.add_argument("--condition", default="c1_react",
                   choices=[c.value for c in Condition])
    p.add_argument("--caps", default="8000,16000,32000,64000")
    p.add_argument("--levels", default=None, help="comma-separated easy,medium,hard")
    p.add_argument("--instance-ids", default=None, help="comma-separated instance ids")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    p.add_argument("--request-timeout", type=float, default=1800.0,
                   help="per-request timeout in seconds. The default client value of "
                        "180 is far too low with prefix caching off: at ~47 tok/s a "
                        "single window-sized generation takes ~11 minutes.")
    p.add_argument("--max-model-len", type=int, default=32768,
                   help="deployment context window; the official sweep must pass the "
                        "served value (32768). Every request is clamped to it.")
    p.add_argument("--out", type=Path, default=Path("results/logs/formal_pilot"))
    p.add_argument("--conditions", default=None,
                   help="only with --write-expected-runs: comma-separated conditions "
                        "to enumerate (default: the --condition value)")
    p.add_argument("--write-expected-runs", type=Path, default=None,
                   help="write the full list of expected run ids and exit without "
                        "running anything. The pre-registered completion criterion: "
                        "analysis may begin only once every id here has a verified "
                        "result on disk.")
    p.add_argument("--dry-run", action="store_true",
                   help="verify every instance and list the work; never call the endpoint")
    return p


def _csv(text: str | None) -> list[str] | None:
    return [t.strip() for t in text.split(",") if t.strip()] if text else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    caps = [int(c) for c in args.caps.split(",") if c.strip()]

    # The official sweep may not run without a declared context window: without it
    # the top rung would ask for ~63.5k tokens against a 32768 window and the
    # endpoint would reject every call at that cap.
    if args.max_model_len is None or args.max_model_len <= 0:
        raise SystemExit(
            "--max-model-len must be a positive number of tokens: the pilot sweep "
            "runs against a real deployment and every call is bounded by its window"
        )

    subset, binning = load_manifests(args.subset, args.binning_manifest)
    boundaries = binning["boundaries"]

    levels, instance_ids = _csv(args.levels), _csv(args.instance_ids)
    items = canonical_items(subset, caps)
    mine = select_items(
        items,
        shard_index=args.shard_index, num_shards=args.num_shards,
        levels=levels, instance_ids=instance_ids,
    )

    if args.write_expected_runs is not None:
        conditions = _csv(args.conditions) or [args.condition]
        # The manifest describes what THIS invocation would run, so it goes through the
        # same `select_items` the run path uses. It previously enumerated the whole
        # subset and silently ignored --instance-ids, --levels and the shard split, which
        # is worse than not offering the filters at all: the file looked authoritative
        # and named runs the command would never have executed.
        expected = [
            {
                "run_id": run_id_for(cond, item.instance_id, item.cap, args.model),
                "condition": cond, "instance_id": item.instance_id,
                "cap": item.cap, "level": item.level,
            }
            for cond in conditions
            for item in mine
        ]
        payload = {
            "schema_version": "expected_runs/1.0",
            "subset_hash": subset["content_hash"],
            "binning_hash": binning["content_hash"],
            "model": args.model,
            "conditions": conditions,
            "caps": sorted(caps),
            # Recorded so a filtered manifest can never be mistaken for the whole subset.
            "filters": {
                "shard_index": args.shard_index, "num_shards": args.num_shards,
                "levels": levels, "instance_ids": instance_ids,
                "subset_instances": sum(len(v) for v in subset["instances"].values()),
            },
            "n_instances": len({item.instance_id for item in mine}),
            "n_expected_runs": len(expected),
            "runs": expected,
        }
        payload["content_hash"] = content_hash(payload)
        args.write_expected_runs.parent.mkdir(parents=True, exist_ok=True)
        with args.write_expected_runs.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"expected runs: {len(expected)} "
              f"({len(conditions)} conditions x {len(sorted(caps))} caps x "
              f"{payload['n_instances']} instances)")
        if levels or instance_ids or args.num_shards > 1:
            print(f"filters applied: {payload['filters']} "
                  f"(subset holds {payload['filters']['subset_instances']} instances)")
        print(f"content_hash {payload['content_hash']}")
        print(f"written: {args.write_expected_runs}")
        return 0

    print("=" * 72)
    print("PILOT SWEEP over the frozen subset -- composition is replayed, never chosen")
    print("=" * 72)
    print(f"subset  : {args.subset}  ({subset['content_hash'][:12]})")
    print(f"binning : {binning['content_hash'][:12]}   grid {subset.get('grid_name')}")
    print(f"condition {args.condition} | caps {sorted(caps)} | "
          f"shard {args.shard_index}/{args.num_shards} | "
          f"max_model_len {args.max_model_len} | timeout {args.request_timeout}s")
    print(f"items: {len(mine)} of {len(items)} in the canonical order")

    # Integrity first, for every instance in this shard, before a single token is spent.
    verified: dict[str, Instance] = {}
    for item in mine:
        if item.instance_id not in verified:
            verified[item.instance_id] = rebuild_and_verify(item.entry, boundaries)
    print(f"verified {len(verified)} distinct instances against the manifest: "
          "id, complexity metric, optimum and level all match")

    args.out.mkdir(parents=True, exist_ok=True)
    todo, done = [], []
    for item in mine:
        path = result_path(args.out, args.condition, item.instance_id, item.cap, args.model)
        if not path.exists():
            todo.append((item, path))
            continue
        # Existence is not completion: prove the file is this run before skipping it.
        verify_completed(
            path,
            run_id=run_id_for(args.condition, item.instance_id, item.cap, args.model),
            instance_id=item.instance_id, condition=args.condition, cap=item.cap,
            subset_hash=subset["content_hash"], binning_hash=binning["content_hash"],
        )
        done.append((item, path))
    print(f"resume: {len(done)} verified complete on disk, {len(todo)} to run")

    if args.dry_run:
        for item, path in todo[:10]:
            print(f"  [{item.index:>4}] {item.level:<7} cap {item.cap:>6} "
                  f"{item.instance_id}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        print("dry run: the endpoint was never contacted, nothing written")
        return 0

    from src.core import DEFAULT_TOKENIZER, LLMClient, QwenTokenizer

    # If the module default ever moves, the pinned constant must move with it --
    # otherwise the recorded revision would describe a different repository.
    if COUNTING_TOKENIZER != DEFAULT_TOKENIZER:
        raise SystemExit(
            f"counting tokenizer mismatch: run_pilot_sweep pins {COUNTING_TOKENIZER!r} "
            f"but src.core.DEFAULT_TOKENIZER is {DEFAULT_TOKENIZER!r}"
        )
    print(f"counting tokenizer: {COUNTING_TOKENIZER}@{COUNTING_TOKENIZER_REVISION}")
    client = LLMClient(
        tokenizer=QwenTokenizer(COUNTING_TOKENIZER, revision=COUNTING_TOKENIZER_REVISION),
        model=args.model, base_url=args.base_url,
        max_model_len=args.max_model_len, timeout=args.request_timeout,
    )
    for n, (item, _) in enumerate(todo, 1):
        _, run = execute_item(
            item=item, instance=verified[item.instance_id], client=client,
            subset=subset, binning=binning, condition=args.condition,
            model=args.model, base_url=args.base_url,
            shard_index=args.shard_index, num_shards=args.num_shards, out_dir=args.out,
            max_model_len=args.max_model_len,
            request_timeout=args.request_timeout,
        )
        rr = run.run_result
        print(f"[{n}/{len(todo)}] {item.level:<7} cap {item.cap:>6} "
              f"sat {rr.score.satisfaction:.2f} tokens {rr.tokens.total:>6} "
              f"{item.instance_id}")

    docs = []
    for item in mine:
        path = result_path(args.out, args.condition, item.instance_id, item.cap, args.model)
        if path.exists():
            docs.append(json.loads(path.read_text(encoding="utf-8")))
    print()
    print("=" * 72)
    for cap in sorted(caps):
        subset_docs = [d for d in docs if d["run_result"]["cap"] == cap]
        if subset_docs:
            print(f"cap {cap:>6}: {json.dumps(summarise(subset_docs), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
