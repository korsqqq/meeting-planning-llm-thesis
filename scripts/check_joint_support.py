# scripts/check_joint_support.py
"""Joint support of density, optimum and higher-order gap. CPU only, read-only.

    python -m scripts.check_joint_support \
        --pool results/calibration/structural \
        --out results/analysis/joint_support

**This check is POST-SPECIFIED and is labelled as such wherever it is reported.** It was
not part of the analysis committed before the pool was read. It exists because a
diagnostic that *was* pre-registered returned a result that changes what the next step has
to control for: the higher-order gap `H = alpha_reachable - O` averages 1.41 at `D = 0` and
2.30 at two conflicting pairs, falling to zero at the top of the range. Instances the
pairwise metric calls conflict-free carry the most interaction it cannot see.

That matters because the research question is about interacting constraints, and `D`
measures only the pairwise part. If the eventual bands were cut on `D` alone, `H` would
move in the opposite direction across them, and any difference between conditions could be
read as an effect of pairwise density or of higher-order interaction moving the other way.

So the question here is narrow: **is there common support in which `D` varies widely while
`O` and `H` are both held comparable?**

`H` becomes a matching / nuisance variable, exactly as `O` already is. It does **not**
become part of the definition of complexity, and no composite score of the form `D + H` is
formed anywhere. The distinction is deliberate: the design tests the effect of pairwise
interaction density, and holds the higher-order part comparable so that it cannot explain
the result.

Nothing here selects a band. There is no Low/Medium/High constant, no threshold, and no
search for the cell or triple of cells that would look best. The support tables are the
input to a decision taken against the rules in THESIS_DECISIONS section 3 and recorded in a
dated amendment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyse_structural_calibration import (  # noqa: E402
    MAX_PAIRS,
    N_PEOPLE,
    load_pool,
    spearman,
)

SCHEMA_VERSION = "joint_support/1.0"


def _cell_summary(g: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs = [r["conflicting_pairs"] for r in g]
    tight = Counter(r["tightness"] for r in g)
    top_t, top_n = tight.most_common(1)[0]
    return {
        "n": len(g),
        "pair_min": min(pairs),
        "pair_max": max(pairs),
        "pair_span": max(pairs) - min(pairs),
        "distinct_pair_counts": len(set(pairs)),
        "D_min": round(min(pairs) / MAX_PAIRS, 4),
        "D_max": round(max(pairs) / MAX_PAIRS, 4),
        "distinct_tightness": len(tight),
        "dominant_tightness": top_t,
        "dominant_tightness_share": round(top_n / len(g), 4),
        "distinct_structures": len({r["travel_structure"] for r in g}),
        # Does the tightness confound survive conditioning on (O, H)? If it does, a band
        # cut inside one cell is still largely a band cut on one generator knob.
        "spearman_D_vs_tightness": spearman([r["D"] for r in g],
                                            [r["tightness"] for r in g]),
        "counts_per_pair_count": dict(sorted(Counter(pairs).items())),
    }


def joint_support(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[(r["oracle_optimum"], r["higher_order_gap_H"])].append(r)

    per_cell = {f"O={o},H={h}": _cell_summary(g) for (o, h), g in sorted(cells.items())}

    h_given_o: dict[str, dict[str, int]] = {}
    by_o: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_o[r["oracle_optimum"]].append(r)
    for o, g in sorted(by_o.items()):
        h_given_o[str(o)] = dict(sorted(Counter(r["higher_order_gap_H"] for r in g).items()))

    return {
        "n": len(rows),
        "spearman_D_vs_H": spearman([r["D"] for r in rows],
                                    [r["higher_order_gap_H"] for r in rows]),
        "spearman_H_vs_optimum": spearman([r["higher_order_gap_H"] for r in rows],
                                          [r["oracle_optimum"] for r in rows]),
        "H_distribution_given_optimum": h_given_o,
        "per_cell": per_cell,
    }


def _f(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    j = doc["joint_support"]

    add("# Joint support of `D`, oracle optimum and higher-order gap\n")
    add("> **This diagnostic is post-specified.** It was not part of the analysis committed "
        "before the calibration pool was read. It is motivated by a result from a "
        "diagnostic that *was* pre-registered: the higher-order gap `H` is largest exactly "
        "where the pairwise density `D` is smallest, so bands cut on `D` alone would move "
        "`H` in the opposite direction and confound the two.\n")
    add("`H` is treated here as a matching variable alongside the oracle optimum `O`. It "
        "does **not** enter the definition of complexity, and no composite of the form "
        "`D + H` is formed. The design tests pairwise interaction density and holds the "
        "higher-order part comparable so that it cannot explain the outcome.\n")
    add("**No band is selected here.** No threshold, no Low/Medium/High constant, and no "
        "search for the cell or triple of cells that would look best.\n")

    add(f"Pool: {j['n']} candidates at `n_people = {N_PEOPLE}`. "
        f"Spearman(`D`, `H`) = **{_f(j['spearman_D_vs_H'])}**; "
        f"Spearman(`H`, optimum) = {_f(j['spearman_H_vs_optimum'])}.\n")

    add("\n## How `H` is distributed at each optimum\n")
    add("| optimum | " + " | ".join(f"H={h}" for h in range(-1, 6)) + " |")
    add("|---" * 8 + "|")
    for o, dist in j["H_distribution_given_optimum"].items():
        cells = " | ".join(str(dist.get(h, dist.get(str(h), 0))) for h in range(-1, 6))
        add(f"| {o} | {cells} |")

    add("\n## Common support: how wide is `D` inside a fixed (optimum, `H`) cell\n")
    add("A cell is usable for the intended contrast only if it holds enough candidates "
        "across a wide enough span of `D`. Both columns are reported; neither is thresholded "
        "here.\n")
    add("| cell | n | pair span | distinct pairs | D min | D max | distinct tightness | dominant share | structures | ρ(D, tightness) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for name, c in j["per_cell"].items():
        add(f"| `{name}` | {c['n']} | {c['pair_span']} | {c['distinct_pair_counts']} | "
            f"{_f(c['D_min'])} | {_f(c['D_max'])} | {c['distinct_tightness']} | "
            f"{_f(c['dominant_tightness_share'])} | {c['distinct_structures']} | "
            f"{_f(c['spearman_D_vs_tightness'])} |")

    add("\n## What this report does not do\n")
    add("- It selects no band and applies no threshold.")
    add("- It does not rank cells or search for the best triple. That search would choose "
        "the design from its own outcome.")
    add("- It does not redefine complexity. `D` remains the axis; `O` and `H` are matching "
        "variables.\n")
    add("The decision this feeds is which of the three outcomes recorded in §3 applies: "
        "common support is adequate and `D` bands are cut with `O` and `H` balanced; `H` "
        "cannot be balanced and the axis is renamed to pairwise interaction density with "
        "`H` reported separately; or `D` and `H` are structurally opposed with no common "
        "support, and the one-dimensional scale is abandoned for a `(D, H)` representation.\n")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path("results/calibration/structural"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/joint_support"))
    args = ap.parse_args(argv)

    rows, meta = load_pool(args.pool)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": "post-specified structural diagnostic motivated by the preregistered H result",
        "selects_bands": False,
        "pool_dir": args.pool.as_posix(),
        "pool_commit": meta.get("git_commit"),
        "joint_support": joint_support(rows),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    write_report(doc, args.out / "report.md")

    j = doc["joint_support"]
    print(f"POST-SPECIFIED check over {j['n']} candidates")
    print(f"Spearman(D, H) = {j['spearman_D_vs_H']}, "
          f"Spearman(H, optimum) = {j['spearman_H_vs_optimum']}")
    wide = [(k, c) for k, c in j["per_cell"].items() if c["distinct_pair_counts"] >= 10]
    print(f"cells with at least 10 distinct densities: {len(wide)}")
    for k, c in sorted(wide, key=lambda kv: -kv[1]["n"])[:8]:
        print(f"  {k:<12} n={c['n']:<5} D {c['D_min']:.3f}-{c['D_max']:.3f} "
              f"({c['distinct_pair_counts']} values, dominant tightness "
              f"{c['dominant_tightness_share']:.2f}, {c['distinct_structures']} structures)")
    print(f"written: {args.out}/summary.json, report.md")
    print("no band selected; decide against the §3 rules and record a dated amendment")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
