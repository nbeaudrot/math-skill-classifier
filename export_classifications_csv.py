"""Export saved classify_skills.py results to a grouped CSV.

Each question is one chunk: a header row with question_number and statement, then one row
per skill with category, skill, importance, new_skill (yes/no), and why starting in column 3.

Examples:
    python export_classifications_csv.py --input output/amc8-2022.json \\
        --model anthropic/claude-haiku-5.5@high --answer-key \\
        --exclude 10 \\
        --out output/skills/amc8-2022-with-key-classification.csv

    python export_classifications_csv.py @sample-problem-manifests/amc8-25questions.txt \\
        --input output/amc8-2022.json --model anthropic/claude-haiku-5.5@high \\
        --out output/skills/amc8-2022-sample.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from classify_skills import (load_problems, match_skills, parse_run_label, reparse_saved,
                             results_dir, split_model_spec)


def problem_number(problem):
    label = problem.get("label")
    if label is not None:
        return str(label)
    known = problem.get("knownAs") or ""
    if "#" in known:
        return known.split("#", 1)[1].split()[0]
    return problem["id"]


def sort_key_for_problem(problem):
    num = problem_number(problem)
    return (0, int(num), num) if num.isdigit() else (1, 0, num)


def load_classification(run_path, problem_id):
    path = run_path / f"{problem_id}.json"
    if not path.exists():
        return None, f"no saved result at {path}"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec = reparse_saved(rec)
    if rec.get("result") is None:
        return None, f"could not parse model reply in {path}"
    return match_skills(rec["result"]), None


def skill_is_new(skill):
    if skill.get("new"):
        return "yes"
    if skill.get("listed") is False:
        return "yes"
    return "no"


def export_rows(problems_by_id, problem_ids, run_path):
    rows = []
    missing = []
    n_questions = 0
    for pid in problem_ids:
        problem = problems_by_id.get(pid)
        if problem is None:
            missing.append(f"{pid}: not in --input")
            continue
        result, err = load_classification(run_path, pid)
        if err:
            missing.append(f"{problem_number(problem)} ({pid}): {err}")
            continue
        qnum = problem_number(problem)
        stmt = problem["details"]["statement"].strip()
        rows.append([qnum, stmt, "", "", "", "", ""])
        n_questions += 1
        for s in result.get("skills") or []:
            rows.append([
                "",
                "",
                s.get("category") or "",
                s.get("skill") or "",
                s.get("importance") or "",
                skill_is_new(s),
                s.get("why") or "",
            ])
    return rows, missing, n_questions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")
    ap.add_argument("problem_ids", nargs="*", metavar="PROBLEM_ID",
                    help="problem IDs to export (default: every problem in --input, sorted by number)")
    ap.add_argument("--input", nargs="+", required=True, help="problem JSON from fetch_problems.py")
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5@high",
                    help="model run folder name (default: anthropic/claude-haiku-5.5@high)")
    ap.add_argument("--answer-key", action="store_true",
                    help="read results from the -with-key folder (classify_skills.py --answer-key)")
    ap.add_argument("--results-dir", default="output/skills",
                    help="parent folder where classify_skills.py saved results")
    ap.add_argument("--run-dir", metavar="PATH",
                    help="exact results folder; overrides --model and --answer-key")
    ap.add_argument("--exclude", action="append", default=[], metavar="N",
                    help="skip question numbers (problem labels), e.g. 10 (repeatable)")
    ap.add_argument("--out", required=True, metavar="PATH", help="CSV file to write")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    problems = load_problems(args.input)
    exclude = {str(x) for x in args.exclude}

    if args.run_dir:
        run_path = Path(args.run_dir)
    else:
        model, effort = split_model_spec(args.model)
        run_path = results_dir(args.results_dir, model, effort, args.answer_key)

    if args.problem_ids:
        ids = list(args.problem_ids)
    else:
        ids = [p["id"] for p in sorted(problems.values(), key=sort_key_for_problem)
               if problem_number(p) not in exclude]

    ids = [pid for pid in ids if problem_number(problems[pid]) not in exclude]

    rows, missing, n_questions = export_rows(problems, ids, run_path)
    if missing:
        for msg in missing:
            print(msg, file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["question_number", "statement", "category", "skill", "importance", "new_skill", "why"]
    n_skill_rows = len(rows) - n_questions
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(rows)

    print(f"Wrote {n_skill_rows} skill row(s) for {n_questions} question(s) from {run_path} to {out}",
          file=sys.stderr)
    if missing and not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
