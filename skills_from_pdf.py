"""Ask an LLM to read a reference PDF and list the skills it covers, in the same format as skills.csv.

The model gets only the PDF, not skills.csv, so the two lists can be compared independently.

Examples:
    # Build a skill list from the MathCounts Playbook
    python skills_from_pdf.py https://monks.scranton.edu/files/courses/ProblemSolving/MathCountsPlaybookBW.pdf

    # Use a local PDF and a different model
    python skills_from_pdf.py playbook.pdf --model anthropic/claude-opus-5 --out skills-other.csv
"""

import argparse
import base64
import csv
import json
import sys
import urllib.request
from pathlib import Path

from classify_skills import KEY_FILE, SKILL_CATEGORIES, parse_json_reply, post_openrouter

DEFAULT_MODEL = "anthropic/claude-opus-5.5"
MAX_TOKENS = 32000

category_list = ", ".join(SKILL_CATEGORIES)

PROMPT = f"""\
The attached PDF is a reference sheet of the math a student needs for middle school math \
competitions such as MATHCOUNTS and AMC 8. Turn it into a list of skills that a coach could \
use to tag competition problems.

Guidelines:
- Cover everything in the PDF, including its test-taking strategy tips.
- Pitch each skill at the level of a teachable topic, e.g. "Counting divisors from the prime \
factorization" or "Area of a trapezoid". Don't make one skill per table row or per individual \
fact (all the divisibility tests are one skill), and don't use broad areas like "Geometry".
- Only include skills the PDF actually covers. Don't add topics it leaves out.
- Give each skill one category from: {category_list}. Use "logic-and-geometry" for geometry \
and logical reasoning, and "test-taking" for strategies like guessing and spotting shortcuts.
- Write a one-sentence description of what the skill involves, specific enough to tell \
similar skills apart.
- Record which section of the PDF the skill comes from, using its numbering (e.g. "2c", "11g").

Respond with JSON only, in this format:
{{
  "skills": [
    {{
      "category": "<one of: {category_list}>",
      "skill": "<short skill name>",
      "description": "<one sentence>",
      "section": "<section number in the PDF>"
    }}
  ]
}}"""


def read_pdf(source):
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as resp:
            return resp.read()
    return Path(source).read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="URL or path of the PDF")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model ID (default {DEFAULT_MODEL})")
    ap.add_argument("--out", default="skills-playbook.csv", help="CSV to write (default skills-playbook.csv)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pdf = read_pdf(args.pdf)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "file", "file": {"filename": Path(args.pdf).name or "document.pdf",
                                  "file_data": "data:application/pdf;base64," + base64.b64encode(pdf).decode()}},
    ]}]
    print(f"Sending {len(pdf) // 1024} KB PDF to {args.model}...", file=sys.stderr)
    resp = post_openrouter({"model": args.model, "messages": messages, "max_tokens": MAX_TOKENS},
                           KEY_FILE.read_text(encoding="utf-8").strip(), timeout=600)
    choice = resp["choices"][0]
    reply = choice["message"].get("content") or ""

    out = Path(args.out)
    raw_path = out.with_suffix(".response.json")
    raw_path.write_text(json.dumps({"model": args.model, "pdf": args.pdf, "prompt": PROMPT, "reply": reply,
                                    "finishReason": choice.get("finish_reason"), "usage": resp.get("usage")},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    skills = parse_json_reply(reply)["skills"]

    bad = [s["skill"] for s in skills if s.get("category") not in SKILL_CATEGORIES]
    if bad:
        print(f"warning: unknown category for {bad}", file=sys.stderr)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["category", "skill", "description", "playbook_section"])
        for s in skills:
            w.writerow([s.get("category"), s.get("skill"), s.get("description"), s.get("section")])

    usage = resp.get("usage") or {}
    print(f"Wrote {len(skills)} skills to {out} (raw reply in {raw_path})", file=sys.stderr)
    print(f"Tokens: {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out; "
          f"cost ${usage.get('cost')}; finish reason {choice.get('finish_reason')}", file=sys.stderr)


if __name__ == "__main__":
    main()
