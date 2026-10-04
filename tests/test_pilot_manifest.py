# tests/test_pilot_manifest.py
"""Selection logic for the pilot manifest -- offline, with a scripted evaluator.

CP-SAT is not exercised here on purpose: what needs locking down is the SELECTION,
which is what makes the pilot pre-registered. A fake evaluator lets a test state
exactly which seeds fail and why, which the real solver cannot be asked to do.

Covered: cell order, first-N-qualifying in ascending order, both rejection reasons and
their rates, seed exhaustion, level binning over the whole pool, hash determinism
across reruns, hash insensitivity to timings, and the round-robin subset selection.
"""

from __future__ import annotations

from scripts.build_pilot_manifest import (
    DEV_GRID,
    Cell,
    build_binning_manifest,
    content_hash,
    grid_cells,
    rejection_reason,
    scan_cell,
    select_pilot_subset,
)
from src.schemas import Level, TravelStructure


def _ok(optimum: int = 3, metric: int = 0, seconds: float = 0.1) -> dict:
    return {"instance_id": "x", "status": "OPTIMAL", "proven_optimal": True,
            "optimum": optimum, "complexity_metric": metric, "solve_seconds": seconds}


def _cell() -> Cell:
    return Cell(4, 0.2, 0.2, "uniform")


# --------------------------------------------------------------------------- #
# Grid enumeration.
# --------------------------------------------------------------------------- #
def test_grid_cells_are_sorted_and_complete() -> None:
    grid = {
        "n_people": [6, 4],
        "tightness": [1.0, 0.2],
        "overlap": [0.2],
        "travel_structure": [TravelStructure.CLUSTERED, TravelStructure.UNIFORM],
    }
    cells = grid_cells(grid)
    assert len(cells) == 2 * 2 * 1 * 2
    assert cells == sorted(cells)  # order cannot depend on how the grid was written
    assert cells[0] == Cell(4, 0.2, 0.2, "clustered")


def test_pilot_grid_size_matches_the_locked_design() -> None:
    from scripts.build_pilot_manifest import PILOT_GRID
    assert len(grid_cells(PILOT_GRID)) == 48


# --------------------------------------------------------------------------- #
# Qualification.
# --------------------------------------------------------------------------- #
def test_rejection_reasons_are_distinguished() -> None:
    assert rejection_reason(_ok()) is None
    assert rejection_reason({**_ok(), "proven_optimal": False}) == "not_proven"
    assert rejection_reason({**_ok(), "optimum": 0}) == "optimum_zero"
    # An unproven optimum is a solver limitation and takes precedence over the value.
    assert rejection_reason({**_ok(), "proven_optimal": False, "optimum": 0}) == "not_proven"


def test_scan_takes_the_first_n_qualifying_in_ascending_order() -> None:
    # Seeds 0 and 2 fail; the accepted set must be exactly 1, 3, 4 -- never a later
    # seed swapped in for a "nicer" one.
    def evaluate(cell: Cell, seed: int) -> dict:
        if seed == 0:
            return {**_ok(), "proven_optimal": False, "status": "UNKNOWN"}
        if seed == 2:
            return {**_ok(), "optimum": 0}
        return _ok(metric=seed)

    scan = scan_cell(_cell(), range(0, 100), 3, evaluate)
    assert [r["seed"] for r in scan["accepted"]] == [1, 3, 4]
    assert [r["seed"] for r in scan["rejected"]] == [0, 2]
    assert [r["reason"] for r in scan["rejected"]] == ["not_proven", "optimum_zero"]
    assert scan["complete"] is True
    assert scan["seed_range_exhausted"] is False
    # Five seeds touched for three acceptances: the scan stops at the third.
    assert scan["n_considered"] == 5
    assert scan["rates"]["n_not_proven"] == 1
    assert scan["rates"]["n_optimum_zero"] == 1
    assert scan["rates"]["not_proven_rate"] == 0.2
    assert scan["rates"]["optimum_zero_rate"] == 0.2


def test_scan_reports_an_exhausted_seed_range_instead_of_silently_short() -> None:
    scan = scan_cell(_cell(), range(0, 4), 3, lambda c, s: {**_ok(), "optimum": 0})
    assert scan["accepted"] == []
    assert scan["complete"] is False
    assert scan["seed_range_exhausted"] is True
    assert scan["n_considered"] == 4


# --------------------------------------------------------------------------- #
# Binning over the whole pool, and hashing.
# --------------------------------------------------------------------------- #
def _metric_by_seed(metrics: dict[int, int]):
    def evaluate(cell: Cell, seed: int) -> dict:
        return _ok(metric=metrics.get(seed, 0), seconds=0.01 * seed)
    return evaluate


def test_levels_are_cut_over_the_whole_pool_not_per_cell() -> None:
    # DEV_GRID has two cells; the metrics 0,0,1,1,5,9 across both must be binned once.
    metrics = {0: 0, 1: 0, 2: 1, 3: 1, 4: 5, 5: 9}
    m = build_binning_manifest(DEV_GRID, "dev", 0, 100, 3, _metric_by_seed(metrics))

    assert m["n_cells"] == 2 and m["n_accepted"] == 6
    # easy = zero conflicts; the positive metrics (1,1,5,9 twice over) split at median.
    assert m["boundaries"]["b1_easy_max"] == 0
    assert m["level_counts"][Level.EASY.value] == 4  # seeds 0 and 1 in both cells
    assert sum(m["level_counts"].values()) == 6
    assert m["incomplete_cells"] == []


def test_hash_is_reproducible_and_ignores_timings() -> None:
    metrics = {0: 0, 1: 2, 2: 7}
    first = build_binning_manifest(DEV_GRID, "dev", 0, 100, 3, _metric_by_seed(metrics))

    # Same inputs, different wall-clock: the hash must not move.
    def slower(cell: Cell, seed: int) -> dict:
        return _ok(metric=metrics.get(seed, 0), seconds=99.0 + seed)
    second = build_binning_manifest(DEV_GRID, "dev", 0, 100, 3, slower)

    assert first["content_hash"] == second["content_hash"]
    # ... but a different accepted pool must move it.
    third = build_binning_manifest(DEV_GRID, "dev", 0, 100, 2, _metric_by_seed(metrics))
    assert third["content_hash"] != first["content_hash"]


def test_hash_covers_the_grid_and_the_seed_range() -> None:
    ev = _metric_by_seed({})
    base = build_binning_manifest(DEV_GRID, "dev", 0, 100, 2, ev)
    moved = build_binning_manifest(DEV_GRID, "dev", 50, 100, 2, ev)
    assert base["content_hash"] != moved["content_hash"]
    assert content_hash({"a": 1}) == content_hash({"a": 1})


# --------------------------------------------------------------------------- #
# Stage 2: subset selection.
# --------------------------------------------------------------------------- #
def _manifest_with(rows: list[dict]) -> dict:
    return {"schema_version": "pilot_manifest/1.0", "grid_name": "test",
            "content_hash": "deadbeef", "pool": rows}


def _row(seed: int, level: Level, n_people: int, travel: str) -> dict:
    return {
        "seed": seed, "instance_id": f"i{seed}-{n_people}-{travel}", "level": level.value,
        "optimum": 3, "complexity_metric": 0,
        "cell": {"n_people": n_people, "tightness": 0.2, "overlap": 0.2,
                 "travel_structure": travel},
        "cell_key": f"n{n_people}-t20-o20-{travel}",
    }


def test_subset_spreads_across_strata_instead_of_draining_one() -> None:
    # 6 instances of one stratum with the LOWEST seeds, 6 of another with higher seeds.
    # A plain ascending-seed cut would take 4 from the first stratum only; round-robin
    # must alternate, so the subset keeps both.
    rows = ([_row(s, Level.EASY, 4, "uniform") for s in range(0, 6)]
            + [_row(s, Level.EASY, 8, "clustered") for s in range(100, 106)])
    subset = select_pilot_subset(_manifest_with(rows), per_level=4)

    picked = subset["instances"][Level.EASY.value]
    assert len(picked) == 4
    assert [p["seed"] for p in picked] == [0, 100, 1, 101]
    assert subset["representation"][Level.EASY.value]["n_people"] == [4, 8]
    assert subset["counts"][Level.EASY.value] == 4
    assert subset["short_levels"] == [Level.MEDIUM.value, Level.HARD.value]


def test_subset_is_deterministic_and_ascending_within_a_stratum() -> None:
    rows = [_row(s, Level.HARD, 6, "uniform") for s in (7, 1, 5, 3)]
    a = select_pilot_subset(_manifest_with(rows), per_level=3)
    b = select_pilot_subset(_manifest_with(list(reversed(rows))), per_level=3)

    assert [p["seed"] for p in a["instances"][Level.HARD.value]] == [1, 3, 5]
    assert a["content_hash"] == b["content_hash"]  # input order must not matter


def test_subset_records_the_manifest_it_came_from() -> None:
    rows = [_row(s, Level.EASY, 4, "uniform") for s in range(3)]
    subset = select_pilot_subset(_manifest_with(rows), per_level=10)
    assert subset["manifest_hash"] == "deadbeef"
    assert subset["per_level_requested"] == 10
    # A level that cannot supply the request is reported, not silently padded.
    assert subset["counts"][Level.EASY.value] == 3
    assert Level.EASY.value in subset["short_levels"]
