"""Put several models' saved skill classifications side by side in one JSON file.

Reads the per-problem results that classify_skills.py saved under output/skills/<model>/.
Each model's skills are written on a single line, essential before helpful, for easy
visual comparison.

Examples:
    python compare_results.py --sample 5 --seed 42 --models qwen/qwen3.8-27b google/gemini-3.8-flash \
        google/gemini-3.8-flash@medium --out output/skills/compare/sample5-seed42.json
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from classify_skills import (answer_choices, build_messages, has_diagram, load_problems, match_skills,
                             parse_run_label, reparse_saved, results_dir)


def skills_for(problem, label, out_dir):
    path = results_dir(out_dir, *parse_run_label(label)) / f"{problem['id']}.json"
    if not path.exists():
        return "no saved result"
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("prompt") != build_messages(problem):
        print(f"warning: {path} was made with a different prompt or skill list", file=sys.stderr)
    if reparse_saved(rec)["result"] is None:
        return "could not parse reply"
    return [{"skill": s.get("skill"), "category": s.get("category"), "importance": s.get("importance")}
            for s in match_skills(rec["result"])["skills"]]


def dumps_with_one_line_arrays(records):
    """Pretty-print, but keep each model's skill array on one line."""
    arrays = []

    def placeholder(value):
        arrays.append(json.dumps(value, ensure_ascii=False))
        return f"@@ARRAY{len(arrays) - 1}@@"

    for r in records:
        r["models"] = {m: placeholder(v) for m, v in r["models"].items()}
    text = json.dumps(records, indent=2, ensure_ascii=False)
    return re.sub(r'"@@ARRAY(\d+)@@"', lambda m: arrays[int(m.group(1))], text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem_ids", nargs="*", metavar="PROBLEM_ID")
    ap.add_argument("--sample", type=int, help="also include N random problems without diagrams (same as classify_skills.py)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model IDs, optionally with a reasoning effort, e.g. google/gemini-3.8-flash@medium")
    ap.add_argument("--input", nargs="+", help="problem JSON file(s) (default: output/*.json)")
    ap.add_argument("--results-dir", default="output/skills", help="where classify_skills.py saved results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems = load_problems(args.input or sorted(Path("output").glob("*.json")))
    ids = list(args.problem_ids)
    if args.sample:
        pool = sorted(pid for pid, p in problems.items() if not has_diagram(p) and pid not in ids)
        ids += random.Random(args.seed).sample(pool, min(args.sample, len(pool)))

    records = []
    for pid in ids:
        p = problems[pid]
        records.append({
            "problemId": pid,
            "knownAs": p.get("knownAs"),
            "statement": p["details"]["statement"],
            "answerChoices": answer_choices(p),
            "models": {m: skills_for(p, m, args.results_dir) for m in args.models},
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_with_one_line_arrays(records) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} problems x {len(args.models)} models to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
