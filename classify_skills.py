"""Ask an LLM which mathematical skills are needed to solve a problem.

Problems are looked up by ID in the JSON files written by fetch_problems.py.
Problems with diagrams (images in the statement or the answer choices) are skipped.

Examples:
    # Classify one problem
    python classify_skills.py X4hTWCLinR4Gs8QLicVw

    # Classify several problems with a different model
    python classify_skills.py X4hTWCLinR4Gs8QLicVw s4HoUcmlPcCxO69Kd67g --model qwen/qwen3.6-27b

    # Classify 5 random problems without diagrams and collect the results in one file
    python classify_skills.py --sample 5 --seed 42 --combined output/skills/sample5-seed42.json

    # Let the model think before answering (results are saved separately from default runs)
    python classify_skills.py --sample 5 --seed 42 --model google/gemini-3.8-flash --reasoning-effort medium

    # Classify the problems listed in a file (one ID per line)
    python classify_skills.py @eval-amc8-25.txt --model google/gemini-3.8-flash

    # Use a friend's self-hosted Qwen server instead of OpenRouter
    python classify_skills.py X4hTWCLinR4Gs8QLicVw --provider mgb

    # Show the prompt without calling the API
    python classify_skills.py X4hTWCLinR4Gs8QLicVw --dry-run
"""

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
# OpenRouter reserves credit for max_tokens up front; without a cap it uses the model's
# maximum (65k+), which a low-balance key can't cover. This includes reasoning tokens.
MAX_TOKENS = 16000
KEY_FILE = Path(__file__).parent / "local-only" / "openrouter_api_key.txt"
SKILLS_FILE = Path(__file__).parent / "skill-lists" / "skills-merged.csv"

# OpenAI-compatible endpoints. A single self-hosted GPU is much slower than OpenRouter, hence the longer timeout.
PROVIDERS = {
    "openrouter": {"url": API_URL, "key_file": KEY_FILE, "default_model": DEFAULT_MODEL, "timeout": 180},
    "mgb": {"url": "https://bifrost.mgb.monster/v1/chat/completions",
            "key_file": Path(__file__).parent / "local-only" / "mgb_qweb_api_key.txt",
            "default_model": "black4/unsloth/Qwen3.8-27B-GGUF:Q4_K_M", "timeout": 900},
}

CATEGORIES = ["arithmetic-and-statistics", "number theory", "algebra", "logic-and-geometry", "counting-and-probability"]
SKILL_CATEGORIES = CATEGORIES + ["test-taking"]


def load_skills(path=SKILLS_FILE):
    with open(path, encoding="utf-8", newline="") as f:
        skills = list(csv.DictReader(f))
    for s in skills:
        if s["category"] not in SKILL_CATEGORIES:
            raise ValueError(f"{path}: skill {s['skill']!r} has unknown category {s['category']!r}")
    return skills


SKILLS = load_skills()
SKILLS_BY_NAME = {s["skill"].lower(): s for s in SKILLS}
category_list = ", ".join(CATEGORIES)
skill_category_list = ", ".join(SKILL_CATEGORIES)
skill_menu = "\n".join(
    f"\n{cat}:\n" + "\n".join(f"- {s['skill']}: {s['description']}" for s in SKILLS if s["category"] == cat)
    for cat in SKILL_CATEGORIES
).strip()

# Literal braces in the JSON example are doubled ({{ }}) because this is an f-string.
# Python reports a brace mistake when the script starts. The line number usually points
# at the mistake, except where noted:
#   stray "}"              -> SyntaxError: f-string: single '}' is not allowed
#   stray "{" in prose     -> SyntaxError: invalid syntax. Perhaps you forgot a comma?
#   "{word}" by mistake    -> NameError: name 'word' is not defined
#   undoubled "{" in JSON  -> SyntaxError: f-string: single '}' is not allowed, reported at
#                             a LATER line (the first "}}" after it), not at the "{" itself.
LLM_PROMPT = f"""\
You are an experienced math competition coach. Given a competition problem, identify the \
mathematical skills a student needs to solve it.

Guidelines:
- Choose skills from the skill list below, using each skill name exactly as written.
- Only if a reasonable solution needs a skill that nothing on the list covers, add a new \
skill with "new": true and a category from: {skill_category_list}. This should be rare.
- Include only skills actually used in a reasonable solution. Usually 2-6 skills.
- Judge skills relative to the competition level given with the problem. Basic skills \
(fractions, percents, unit conversions, mental math) are often taken for granted \
in advanced competitions, but list them when the problem genuinely exercises them, \
especially on middle school contests. Don't list routine steps that any student at that \
level does without thinking.
- Base the skills on the solution a strong student at that level would most likely find, \
not on the most general or sophisticated method. For example, on a middle school contest, \
don't describe a solution in terms of modular arithmetic when students would just add \
the numbers or use a divisibility rule.
- Mark each skill with one of three levels of importance:
  - "essential": the problem can't reasonably be solved without it.
  - "time-saving": not required, but a big shortcut, turning a long or error-prone \
solution into a short one.
  - "helpful": makes the problem somewhat easier, or offers another way to check the answer.
- For multiple choice problems, look at the answer choices and consider the test-taking \
skills on the list. Include them only when the choices actually make them useful. In \
particular, list "Eliminating answer choices" only when ruling out choices clearly saves \
work compared with solving the problem directly, not when the problem has to be solved \
anyway and the matching choice is then picked.
- Do not solve the problem in detail; a one-sentence justification per skill is enough.
- Also choose the single overall category that best describes the problem as a whole, \
based on its main mathematical content.

Skill list, grouped by category:

{skill_menu}

Respond with JSON only, in this format:
{{
  "summary": "<one sentence describing what the problem asks>",
  "overall_category": "<one of: {category_list}>",
  "skills": [
    {{
      "skill": "<skill name from the list>",
      "importance": "essential" | "time-saving" | "helpful",
      "why": "<one sentence on how the skill is used in this problem>"
    }}
  ]
}}
For a new skill not on the list, also include "new": true and "category"."""


def load_problems(paths):
    problems = {}
    for path in paths:
        for rec in json.loads(Path(path).read_text(encoding="utf-8")):
            for p in rec["problems"]:
                problems[p["id"]] = {**p, "examTitle": rec["title"],
                                     "examType": rec["exam"].get("examType")}
    return problems


def has_diagram(problem):
    d = problem.get("details", {})
    if d.get("image") or d.get("optionImages"):
        return True
    return any(part.get("kind") == "image" for part in d.get("statementParts") or [])


def answer_choices(problem):
    opts = problem["details"].get("options") or {}
    return {k: f"${opts[k]}$" for k in sorted(opts)}


def build_user_prompt(problem):
    lines = [f"Source: {problem.get('knownAs') or problem['examTitle']}"]
    if problem.get("examType"):
        lines.append(f"Competition level: {problem['examType']}")
    lines += ["", "Problem:", problem["details"]["statement"].strip()]
    choices = answer_choices(problem)
    if choices:
        lines += ["", "Answer choices:"]
        lines += [f"({k}) {v}" for k, v in choices.items()]
    return "\n".join(lines)


def call_llm(model, messages, api_key, reasoning_effort=None, provider="openrouter"):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        # OpenRouter's own field; other OpenAI-compatible servers use OpenAI's top-level field.
        if provider == "openrouter":
            body["reasoning"] = {"effort": reasoning_effort}
        else:
            body["reasoning_effort"] = reasoning_effort
    p = PROVIDERS[provider]
    return post_openrouter(body, api_key, timeout=p["timeout"], url=p["url"])


def run_label(model, reasoning_effort=None):
    """How a model + effort setting is named in folders and comparisons, e.g. google/gemini-3.8-flash@medium."""
    return f"{model}@{reasoning_effort}" if reasoning_effort else model


def parse_run_label(label):
    model, _, effort = label.partition("@")
    return model, effort or None


def results_dir(out_dir, model, reasoning_effort=None):
    return Path(out_dir) / re.sub(r"[^A-Za-z0-9.@-]+", "_", run_label(model, reasoning_effort))


def post_openrouter(body, api_key, retries=3, timeout=180, url=API_URL):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if attempt == retries - 1 or e.code < 500 and e.code != 429:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)


def parse_json_reply(text):
    """Parse the model's reply, tolerating ```json fences or text around the object."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    text = m.group() if m else text
    # Models often write LaTeX like \pm or \frac with a single backslash. \pm is invalid JSON, and \frac
    # silently parses as a form feed + "rac", so double any lone backslash that starts a LaTeX command.
    text = re.sub(r"(?<!\\)\\(?=[A-Za-z]{2,})", r"\\\\", text)
    return json.loads(text)


def reparse_saved(rec):
    """Retry parsing a saved reply that an older version of parse_json_reply couldn't handle."""
    if rec["result"] is None and rec.get("rawReply"):
        try:
            rec["result"] = parse_json_reply(rec["rawReply"])
        except (json.JSONDecodeError, AttributeError):
            pass
    return rec


def build_messages(problem):
    return [{"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": build_user_prompt(problem)}]


IMPORTANCE_ORDER = {"essential": 0, "time-saving": 1, "helpful": 2}


def match_skills(result):
    """Use the skill list's exact name and category, mark anything not on the list as unlisted,
    and order skills essential first (keeping the model's order within each group)."""
    result["skills"] = sorted(result.get("skills", []),
                              key=lambda s: IMPORTANCE_ORDER.get(str(s.get("importance")).lower(), 3))
    for s in result["skills"]:
        known = SKILLS_BY_NAME.get(str(s.get("skill", "")).strip().lower())
        if known:
            s["skill"], s["category"], s["listed"] = known["skill"], known["category"], True
            s.pop("new", None)
        else:
            s["listed"] = False
    return result


def classify(problem, model, api_key, reasoning_effort=None, provider="openrouter"):
    messages = build_messages(problem)
    resp = call_llm(model, messages, api_key, reasoning_effort, provider)
    choice = resp["choices"][0]
    reply = choice["message"].get("content") or ""
    try:
        result = match_skills(parse_json_reply(reply))
    except (json.JSONDecodeError, AttributeError):
        result = None
    return {
        "problemId": problem["id"],
        "knownAs": problem.get("knownAs"),
        "model": model,
        "reasoningEffort": reasoning_effort,
        "result": result,
        "rawReply": reply,
        "finishReason": choice.get("finish_reason"),
        "usage": resp.get("usage"),
        "prompt": messages,
    }


def print_result(rec):
    print(f"\n=== {rec['knownAs']} ({rec['problemId']}) ===")
    res = rec["result"]
    if res is None:
        print(f"Could not parse model reply as JSON (finish reason: {rec['finishReason']}):\n"
              + rec["rawReply"])
        return
    print(res.get("summary", ""))
    print(f"Overall category: {res.get('overall_category')}")
    for s in res.get("skills", []):
        flag = "" if s.get("listed") else "  ** NOT ON SKILL LIST **"
        print(f"  [{s.get('importance', '?'):11}] {s.get('skill')}  ({s.get('category')}){flag}")
        print(f"                {s.get('why')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")
    ap.add_argument("problem_ids", nargs="*", metavar="PROBLEM_ID",
                    help="problem IDs, or @FILE to read them from a file with one ID per line")
    ap.add_argument("--sample", type=int, help="also classify N random problems without diagrams")
    ap.add_argument("--seed", type=int, help="random seed for --sample")
    ap.add_argument("--input", nargs="+", help="problem JSON file(s) (default: output/*.json)")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="openrouter",
                    help="which API to call (default openrouter; mgb is a friend's self-hosted Qwen server)")
    ap.add_argument("--model", help=f"model ID (default: the provider's default, {DEFAULT_MODEL} for openrouter)")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"],
                    help="ask the model to think before answering (default: the model's own default)")
    ap.add_argument("--out-dir", default="output/skills", help="where to save results (default output/skills)")
    ap.add_argument("--combined", metavar="PATH",
                    help="also write one JSON array with each problem, its answer choices, and the model's result")
    ap.add_argument("--force", action="store_true",
                    help="call the API even if a saved result with the same prompt exists")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt without calling the API")
    args = ap.parse_args()
    provider = PROVIDERS[args.provider]
    args.model = args.model or provider["default_model"]
    # The Windows console defaults to cp1252, which can't print characters like the square root sign.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    problems = load_problems(args.input or sorted(Path("output").glob("*.json")))
    ids = list(args.problem_ids)
    if args.sample:
        pool = sorted(pid for pid, p in problems.items() if not has_diagram(p) and pid not in ids)
        ids += random.Random(args.seed).sample(pool, min(args.sample, len(pool)))
    if not ids:
        ap.error("give at least one PROBLEM_ID or use --sample")
    api_key = None if args.dry_run else provider["key_file"].read_text(encoding="utf-8").strip()
    out_dir = results_dir(args.out_dir, args.model, args.reasoning_effort)
    combined = []

    for pid in ids:
        problem = problems.get(pid)
        if problem is None:
            print(f"{pid}: not found in input files", file=sys.stderr)
            continue
        if has_diagram(problem):
            print(f"{pid}: skipped (has a diagram)", file=sys.stderr)
            continue
        if args.dry_run:
            print(f"\n=== {pid} ===\n--- system ---\n{LLM_PROMPT}\n--- user ---\n{build_user_prompt(problem)}")
            continue
        out_path = out_dir / f"{pid}.json"
        saved = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
        if (saved and saved.get("prompt") == build_messages(problem)
                and saved.get("reasoningEffort") == args.reasoning_effort and not args.force
                and reparse_saved(saved)["result"] is not None):
            rec = saved
            match_skills(rec["result"])
            print(f"\n(using saved result {out_path})", file=sys.stderr)
        else:
            rec = classify(problem, args.model, api_key, args.reasoning_effort, args.provider)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print_result(rec)
        combined.append({
            "problemId": pid,
            "knownAs": problem.get("knownAs"),
            "statement": problem["details"]["statement"],
            "answerChoices": answer_choices(problem),
            "llmResult": rec["result"] if rec["result"] is not None else rec["rawReply"],
        })

    if not args.dry_run:
        print(f"\nSaved results to {out_dir}", file=sys.stderr)
        if args.combined:
            path = Path(args.combined)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote {len(combined)} results to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
