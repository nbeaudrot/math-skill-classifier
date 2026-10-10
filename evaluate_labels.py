"""Compare models' skill classifications with the hand labels in the categorization spreadsheet.

The spreadsheet tags each problem with up to three category/subcategory pairs. RAW_TO_SKILLS
translates each subcategory into the skill(s) on the classifier's skill list that it corresponds to.

Examples:
    # Print 25 random labeled AMC 8 problem IDs without diagrams (to pass to classify_skills.py)
    python evaluate_labels.py --sample 25 --seed 1 --print-ids

    # Score saved results for those problems
    python evaluate_labels.py --sample 25 --seed 1 --models qwen/qwen3.8-27b google/gemini-3.5-flash-lite@medium \
        --out output/skills/compare/labeled-sample25-seed1.json

    # Score a fixed list of problems (one ID per line)
    python evaluate_labels.py @eval-amc8-25.txt --models qwen/qwen3.8-27b --out output/skills/compare/eval-amc8-25.json
"""

import argparse
import collections
import csv
import random
import re
import sys
from pathlib import Path

from classify_skills import SKILLS_BY_NAME, answer_choices, has_diagram, load_problems
from compare_results import dumps_with_one_line_arrays, skills_for

LABELS_FILE = Path(__file__).parent / "local-only" / "Contest Math Question Categorization - Sheet1.csv"
SHEET_TEST_TO_EXAM_TYPE = {"AMC 8": "AMC-8"}

# Category-column typos in the spreadsheet.
CATEGORY_FIXES = {"algbera": "algebra", "algebria": "algebra", "arithmeti": "arithmetic",
                  "counring": "counting & probability", "geoemtry": "geometry", "logic puzzles": "logic puzzle",
                  "`": "arithmetic", "3d geometry": "geometry"}

# Subcategory (lowercased) -> acceptable skills. "<blank:category>" is a tag with no subcategory;
# "<category>sub" is for subcategories whose meaning depends on the category.
RAW_TO_SKILLS = {}
for skills, keys in [
    (["Regrouping and distributing", "Simplify before you multiply"],
     ["addition", "subtraction", "multiplication", "division", "<blank:arithmetic>"]),
    (["Order of operations"], ["order of operations", "expressions"]),
    (["Fractions", "Simplify before you multiply"], ["fractions", "<blank:fractions>"]),
    (["Decimals"], ["decimals"]),
    (["Percentages"], ["percentages", "percents"]),
    (["Ratios"], ["ratios"]),
    (["Rates"], ["rates", "rate", "distance"]),
    (["Unit conversions"], ["unit conversions", "<blank:unit conversions>"]),
    (["Time and clocks"], ["time", "clocks", "clockface angles"]),
    (["Units digit", "Checking the ones digit"], ["units digit", "unit digit"]),
    (["Comparing and ordering numbers"], ["<arithmetic>inequality", "number line"]),
    (["Mean (average)"], ["average", "averages", "mean"]),
    (["Weighted averages"], ["weighted average"]),
    (["Median"], ["median"]),
    (["Mode"], ["mode"]),
    (["Range"], ["range"]),
    (["Geometric and harmonic means"], ["harmonic mean", "geometric mean"]),
    (["Reading tables and charts"], ["data table", "bar chart", "histogram", "line chart", "stem and leaf",
                                     "dot plot", "scatter plot", "tally"]),
    (["Constructing and solving a single equation of one variable"], ["<blank:algebra>", "combining like terms"]),
    (["Systems of equations"], ["systems of equations"]),
    (["Proportional equations"], ["proportional equations"]),
    (["Inequalities"], ["<algebra>inequality", "inequalities"]),
    (["Absolute value"], ["absolute value"]),
    (["Exponents", "Powers of small numbers"], ["exponents"]),
    (["Square roots"], ["square roots"]),
    (["Factorials"], ["factorials"]),
    (["Functions and defined operations"], ["functions", "operations"]),
    (["Arithmetic sequences"], ["arithmetic sequences", "arithmetic sequenes"]),
    (["Geometric sequences and series"], ["geometric sequences", "geometric sequence", "geometry sequences",
                                          "geometric series", "geometry series"]),
    (["Patterns, periodic, and recursive sequences"], ["sequences", "fibbonacci sequences", "recurrence relations"]),
    (["Telescoping sequences"], ["telescoping sequences"]),
    (["Factoring polynomials"], ["factoring", "factoring polynomials", "polynomial factoring", "special factoring",
                                 "polynomials"]),
    (["Difference of squares"], ["difference of squares"]),
    (["Simon's Favorite Factoring Trick"], ["sfft"]),
    (["Symmetric expressions"], ["symmetric expressions", "symmetric expresions"]),
    (["Finding an expression without solving for the variables"], ["sneaky stuff"]),
    (["Quadratic equations"], ["quadratic equation", "quadratic equations"]),
    (["Vieta's formulas"], ["vieta's formula"]),
    (["Coordinate geometry"], ["coordinate plane"]),
    (["Slopes and equations of lines", "Geometric Relationships of slopes"], ["line slopes",
                                                                             "slope of perpendicular lines"]),
    (["Divisibility rules"], ["divisibility rules", "divisbility rules", "divisibility", "divisbility"]),
    (["Factors"], ["factors"]),
    (["Counting and summing divisors"], ["counting factors", "sum of factors"]),
    (["Multiples", "Recognizing multiples"], ["multiples"]),
    (["Primes"], ["primes"]),
    (["Prime factorization"], ["prime factorization", "prime factoriziation"]),
    (["GCD and LCM"], ["gcd", "lcm"]),
    (["Perfect squares and cubes"], ["perfect squares", "cubes"]),
    (["Remainders and modular arithmetic"], ["remainders", "fermat's little theorem"]),
    (["Chinese Remainder Theorem"], ["chinese remainder theorem"]),
    (["Number bases"], ["alternate bases"]),
    (["Parity"], ["parity"]),
    (["Chicken McNugget Theorem"], ["chicken mcnugget theorem"]),
    (["Prime powers in factorials"], ["factorial exponent counting"]),
    (["Number theory casework", "Casework"], ["<number theory>casework"]),
    (["Casework"], ["<counting & probability>casework"]),
    (["Combinations"], ["combinations", "subsets"]),
    (["Permutations"], ["permutations", "circular counting"]),
    (["Complementary counting"], ["complementary counting", "complimentary counting", "complementary probability"]),
    (["Inclusion-exclusion and Venn diagrams"], ["pie", "venn diagrams", "set theory", "<blank:set theory>"]),
    (["Avoiding off-by-one errors"], ["off by one errors"]),
    (["Path counting"], ["path counting"]),
    (["Balls and buckets"], ["balls & buckets"]),
    (["Using symmetry"], ["<counting & probability>symmetry", "symmetric probability", "<logic>symmetry"]),
    (["Pigeonhole principle"], ["pigeon hole principle"]),
    (["Finding a pattern from small cases"], ["engineer's induction", "engineer's inductions"]),
    (["Basic probability"], ["<blank:counting & probability>"]),
    (["Conditional probability"], ["conditional probability"]),
    (["Compound and probability"], ["compound probability", "sampling"]),
    (["Expected value"], ["expected value"]),
    (["Geometric probability"], ["geometric probability"]),
    (["Basic shapes"], ["rectangle area", "triangle area", "circle area", "circle areas", "circle perimeter",
                        "perimeter", "rectangle perimeter", "polygon perimeter"]),
    (["Trapezoid and parallelogram area"], ["trapezoid area", "trapzeoid area"]),
    (["Rhombus and kite area"], ["rhombus area"]),
    (["Composite area", "Basic shapes"], ["area"]),
    (["Area on a grid"], ["lattice area"]),
    (["Circle properties"], ["circles", "circle radius", "circle chords"]),
    (["Tangent lines"], ["tangent lines"]),
    (["Pythagorean theorem"], ["pythagorean theorem"]),
    (["Special right triangles"], ["special triangles", "special right triangles"]),
    (["Similar figures and proportions"], ["similar triangles", "similar shapes & proportions",
                                           "similar shapes & proportionality", "simiplar shapes & proportionality"]),
    (["Angle chasing"], ["angle chasing"]),
    (["Triangle inequality"], ["triangle inequality"]),
    (["Triangle properties"], ["triangles", "triangle geometry"]),
    (["Polygon properties"], ["hexagons", "counting diagonals"]),
    (["Midpoints and centroids"], ["midpoints", "centroid"]),
    (["Angle bisector theorem"], ["angle bisector theorem"]),
    (["Mass points"], ["mass points"]),
    (["Power of a point"], ["power of a point"]),
    (["Cyclic quadrilaterals"], ["cyclic quadrilaterals"]),
    (["Volume of solids"], ["cube volume", "prism volume", "cylinder volume", "pyramid volume", "cone volume",
                            "sphere volume"]),
    (["Surface area of solids"], ["cube surface area", "prism surface area", "sphere surface area"]),
    (["Properties of polyhedra"], ["cube properties", "cube rotations", "prism properties", "tetrahedron",
                                   "tetrahedron properties"]),
    (["Folding and unfolding"], ["folding"]),
    (["Tiling"], ["tiling"]),
    (["Reflections and symmetry"], ["lines of symmetry", "reflections", "<geometry>symmetry"]),
    (["Shortest paths in networks"], ["dijskstra's algorithm"]),
    (["Estimation"], ["estimation", "<blank:estimation>", "rounding"]),
]:
    for k in keys:
        RAW_TO_SKILLS[k] = skills
CONTEXTUAL = {"inequality", "symmetry", "casework"}

# Corrections to the spreadsheet's tags for specific problems: problem ID -> [(acceptable skills, label)].
LABEL_OVERRIDES = {
    "wbzSzvebtEvpqBcu6aCJ": [(["Quadratic equations"], "Algebra")],  # 2019 AMC 8 #20, (x^2 - 5)^2 = 16
}

missing = sorted({s for v in RAW_TO_SKILLS.values() for s in v if s.lower() not in SKILLS_BY_NAME})
if missing:
    raise ValueError(f"RAW_TO_SKILLS refers to skills not on the skill list: {missing}")


def translate(category, sub):
    """Return (acceptable skills, readable label) for one spreadsheet tag; skills is empty if untranslatable."""
    raw_cat = category.strip().lower()
    cat = CATEGORY_FIXES.get(raw_cat, raw_cat)
    s = sub.strip().lower()
    if not s:
        key = f"<blank:{raw_cat}>" if f"<blank:{raw_cat}>" in RAW_TO_SKILLS else f"<blank:{cat}>"
    elif s in CONTEXTUAL:
        key = f"<{cat}>{s}"
    else:
        key = s
    label = f"{category.strip()} / {sub.strip()}" if sub.strip() else category.strip()
    return RAW_TO_SKILLS.get(key, []), label


def load_labels(problems, path=LABELS_FILE):
    """Map problem ID -> list of (acceptable skills, readable label) for every spreadsheet row with a matching problem."""
    by_exam = {(p.get("examType"), p["examTitle"][:4], str(p.get("label"))): pid for pid, p in problems.items()}
    labels, unmatched_rows = {}, 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            exam_type = SHEET_TEST_TO_EXAM_TYPE.get(row["Test"].strip())
            pid = by_exam.get((exam_type, row["Year"].strip(), row["Question #"].strip()))
            if exam_type is None:
                continue
            if pid is None:
                unmatched_rows += 1
                continue
            tags = [translate(row[f"Category{i}"] or "", row[f"Subcategory{i}"] or "")
                    for i in (1, 2, 3) if (row[f"Category{i}"] or row[f"Subcategory{i}"] or "").strip()]
            labels[pid] = LABEL_OVERRIDES.get(pid, tags)
    return labels, unmatched_rows


def pick(labels, problems, n, seed):
    pool = sorted(pid for pid in labels if not has_diagram(problems[pid]))
    return random.Random(seed).sample(pool, min(n, len(pool))), len(pool)


def score(ids, labels, problems, model, results_dir):
    stats = collections.Counter()
    for pid in ids:
        found = skills_for(problems[pid], model, results_dir)
        if isinstance(found, str):
            stats["missing"] += 1
            continue
        yours = {s for skills, _ in labels[pid] for s in skills}
        names = {s["skill"] for s in found}
        stats["skills"] += len(found)
        stats["essential"] += sum(s["importance"] == "essential" for s in found)
        stats["time_saving"] += sum(s["importance"] == "time-saving" for s in found)
        stats["essential_not_yours"] += sum(s["importance"] == "essential" and s["skill"] not in yours for s in found)
        for skills, _ in labels[pid]:
            if skills:
                stats["your_tags"] += 1
                stats["your_tags_found"] += bool(names & set(skills))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")
    ap.add_argument("problem_ids", nargs="*", metavar="PROBLEM_ID",
                    help="problems to score instead of a random sample, or @FILE with one ID per line")
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--print-ids", action="store_true", help="print the chosen problem IDs and stop")
    ap.add_argument("--models", nargs="+", help="model IDs, optionally with a reasoning effort (model@medium)")
    ap.add_argument("--input", nargs="+", help="problem JSON file(s) (default: output/*.json)")
    ap.add_argument("--results-dir", default="output/skills")
    ap.add_argument("--out", help="also write a side-by-side JSON with your labels and each model's skills")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    problems = load_problems(args.input or sorted(Path("output").glob("*.json")))
    labels, unmatched_rows = load_labels(problems)
    if args.problem_ids:
        unlabeled = [pid for pid in args.problem_ids if pid not in labels]
        if unlabeled:
            ap.error(f"no spreadsheet labels for: {' '.join(unlabeled)}")
        ids, pool_size = args.problem_ids, len(args.problem_ids)
    else:
        ids, pool_size = pick(labels, problems, args.sample, args.seed)
    if args.print_ids:
        print("\n".join(ids))
        print(f"{len(labels)} labeled problems found ({unmatched_rows} spreadsheet rows had no matching problem); "
              f"{pool_size} without diagrams; picked {len(ids)}", file=sys.stderr)
        return
    if not args.models:
        ap.error("give --models or --print-ids")

    untranslated = collections.Counter(label for pid in ids for skills, label in labels[pid] if not skills)
    if untranslated:
        print(f"Your tags with no matching skill (can't be matched): {dict(untranslated)}\n")

    print(f"{'model':40} {'skills':>6} {'essential':>10} {'% essential':>12} {'% time-saving':>14} "
          f"{'essential not yours':>20} {'your tags found':>16}")
    for model in args.models:
        s = score(ids, labels, problems, model, args.results_dir)
        pct = 100 * s["essential"] / s["skills"] if s["skills"] else 0
        pct_ts = 100 * s["time_saving"] / s["skills"] if s["skills"] else 0
        found = f"{s['your_tags_found']}/{s['your_tags']}"
        note = f"  ({s['missing']} problems without results)" if s["missing"] else ""
        print(f"{model:40} {s['skills']:>6} {s['essential']:>10} {pct:>11.0f}% {pct_ts:>13.0f}% "
              f"{s['essential_not_yours']:>20} {found:>16}{note}")

    if args.out:
        records = []
        for pid in ids:
            p = problems[pid]
            yours = [{"label": label, "skills": skills} for skills, label in labels[pid]]
            records.append({
                "problemId": pid,
                "knownAs": p.get("knownAs"),
                "statement": p["details"]["statement"],
                "answerChoices": answer_choices(p),
                "models": {"your labels": yours, **{m: skills_for(p, m, args.results_dir) for m in args.models}},
            })
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dumps_with_one_line_arrays(records) + "\n", encoding="utf-8")
        print(f"\nWrote side-by-side comparison to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
