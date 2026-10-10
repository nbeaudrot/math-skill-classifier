"""Fetch math competition problems from the VentiMath catalog.

Examples:
    # List all AMC-8 exams without downloading problems
    python fetch_problems.py --exam-type AMC-8 --list

    # Download problems from 5 randomly chosen AMC-8 exams
    python fetch_problems.py --exam-type AMC-8 --sample 5 --seed 42

    # Download all AIME exams from 2015-2020
    python fetch_problems.py --competition aime --year-from 2015 --year-to 2020

    # Filter on any exam field
    python fetch_problems.py --where calculatorPolicy=not_allowed --exam-type AMC-8
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://us-central1-ventimath.cloudfunctions.net/catalog"
BATCH_SIZE = 50


def get_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def fetch_exams():
    return get_json(f"{BASE_URL}/exams")


def fetch_problems(ids):
    problems = {}
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        url = f"{BASE_URL}/problems?ids=" + urllib.parse.quote(",".join(batch), safe=",")
        for p in get_json(url):
            problems[p["id"]] = p
    return problems


def exam_year(exam):
    """Best-effort numeric year: the `year` field, else the start of `season`."""
    for key in ("year", "season"):
        m = re.match(r"\d{4}", str(exam.get(key) or ""))
        if m:
            return int(m.group())
    return None


def problem_label(ref):
    return str(ref.get("label") or ref.get("problemNumber") or "")


def label_sort_key(label):
    m = re.match(r"(\d+)", label)
    return (0, int(m.group(1)), label) if m else (1, 0, label)


def matches(exam, args):
    def eq(field, value):
        return value is None or str(exam.get(field) or "").lower() == value.lower()

    if not (eq("examType", args.exam_type) and eq("competition", args.competition)
            and eq("event", args.event) and eq("season", args.season)):
        return False
    if args.exam_id and exam["id"] not in args.exam_id:
        return False
    if args.name and args.name.lower() not in exam.get("name", "").lower():
        return False
    year = exam_year(exam)
    if args.year is not None and year != args.year:
        return False
    if args.year_from is not None and (year is None or year < args.year_from):
        return False
    if args.year_to is not None and (year is None or year > args.year_to):
        return False
    for cond in args.where:
        field, _, value = cond.partition("=")
        if not eq(field, value):
            return False
    return len(exam.get("problems", [])) >= args.min_problems


def exam_title(exam):
    year = exam_year(exam)
    name = exam.get("name", exam["id"])
    return name if year is None or str(year) in name else f"{year} {name}"


def build_exam_record(exam, problems):
    refs = sorted(exam.get("problems", []), key=lambda r: label_sort_key(problem_label(r)))
    out = []
    for ref in refs:
        p = problems.get(ref["problemId"])
        if p is None:
            print(f"  warning: problem {ref['problemId']} not returned by API", file=sys.stderr)
            continue
        out.append({"label": problem_label(ref), **p})
    meta = {k: v for k, v in exam.items() if k != "problems"}
    return {"title": exam_title(exam), "exam": meta, "problems": out}


def to_markdown(records):
    lines = []
    for rec in records:
        lines += [f"# {rec['title']}", "", f"Exam ID: `{rec['exam']['id']}`", ""]
        for p in rec["problems"]:
            d = p.get("details", {})
            lines += [f"## Problem {p['label']}", "", d.get("statement", "").strip(), ""]
            img = d.get("image")
            if img and img.get("url"):
                lines += [f"![diagram]({img['url']})", ""]
            opts = d.get("options")
            if opts:
                for k in sorted(opts):
                    lines.append(f"- **({k})** {opts[k]}")
                lines.append("")
        lines.append("")
    return "\n".join(lines)


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam-type", help='e.g. "AMC-8", "AMC-10", "Math Is Cool"')
    ap.add_argument("--competition", help='e.g. "AMC", "aime", "imo", "usamo"')
    ap.add_argument("--event", help='e.g. "day1", "I", "individual"')
    ap.add_argument("--season", help='exact season string, e.g. "2011-12"')
    ap.add_argument("--year", type=int)
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--name", help="case-insensitive substring of exam name")
    ap.add_argument("--exam-id", action="append", default=[], help="specific exam ID (repeatable)")
    ap.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE",
                    help="match any exam field exactly (case-insensitive, repeatable)")
    ap.add_argument("--min-problems", type=int, default=1, help="skip exams with fewer problems (default 1)")
    ap.add_argument("--sample", type=int, help="randomly choose N matching exams")
    ap.add_argument("--seed", type=int, help="random seed for --sample")
    ap.add_argument("--list", action="store_true", help="only list matching exams")
    ap.add_argument("--format", choices=["json", "md", "both"], default="json",
                    help="output format (default json)")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--out-name", help="base filename for outputs (default derived from filters)")
    args = ap.parse_args()

    exams = [e for e in fetch_exams() if matches(e, args)]
    exams.sort(key=lambda e: (exam_year(e) or 0, e.get("name", "")))
    if args.sample is not None and args.sample < len(exams):
        exams = sorted(random.Random(args.seed).sample(exams, args.sample),
                       key=lambda e: (exam_year(e) or 0, e.get("name", "")))

    print(f"{len(exams)} matching exam(s)", file=sys.stderr)
    if args.list or not exams:
        for e in exams:
            print(f"{e['id']}  {exam_title(e)}  ({len(e.get('problems', []))} problems)")
        return

    ids = list(dict.fromkeys(r["problemId"] for e in exams for r in e.get("problems", [])))
    print(f"Fetching {len(ids)} problems...", file=sys.stderr)
    problems = fetch_problems(ids)
    records = [build_exam_record(e, problems) for e in exams]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.out_name or slugify("-".join(
        str(v) for v in (args.exam_type, args.competition, args.year, args.name) if v) or "problems")
    if args.sample:
        base += f"-sample{args.sample}" + (f"-seed{args.seed}" if args.seed is not None else "")
    written = []
    if args.format in ("json", "both"):
        path = out_dir / f"{base}.json"
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(path)
    if args.format in ("md", "both"):
        path = out_dir / f"{base}.md"
        path.write_text(to_markdown(records), encoding="utf-8")
        written.append(path)
    for r in records:
        print(f"  {r['title']}: {len(r['problems'])} problems", file=sys.stderr)
    print("Wrote " + " and ".join(str(p) for p in written), file=sys.stderr)


if __name__ == "__main__":
    main()
