"""Ask an LLM to write student-friendly solutions to competition problems.

Uses the same problem files, problem selection, providers, and manifests as classify_skills.py.
Defaults to Claude Haiku 5.5 on OpenRouter with xhigh reasoning. Problems with diagrams are skipped.

Examples:
    # Solve 5 random problems without diagrams and collect the solutions in one Markdown file
    python solve_problems.py --sample 5 --seed 1 --input output/amc-8-2011-2025.json --combined output/solutions/sample5-seed1.md

    # Solve the problems listed in a file (one ID per line)
    python solve_problems.py @sample-problem-lists/amc8-25questions.txt --input output/amc-8-2011-2025.json

    # Give the model the correct answers from an answer key
    python solve_problems.py @sample-problem-lists/amc8-2022-5questions.txt --input output/amc-8-2011-2025.json --answer-key local-only/aops-answer-keys/2022_amc_8_answer_key.json

    # Show the prompt without calling the API
    python solve_problems.py IBXoa1RxJR1VYlnL4MM7 --dry-run
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from classify_skills import MAX_TOKENS, PROVIDERS, manifest_argv, post_openrouter, results_dir
from models import load_problems

DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
DEFAULT_REASONING_EFFORT = "xhigh"

LLM_PROMPT = """\
You are an experienced math coach writing solutions to math competition problems for middle \
school students.

Guidelines:
- Write a correct solution that a strong middle school student could follow, and could \
have come up with during the contest without a calculator.
- Use only math that an advanced middle school student knows. Do not use trigonometry, \
complex numbers, calculus, or other high school or college math, even when it would be \
shorter. Prefer a clear elementary argument over a clever formula.
- Write at an 8th to 10th grade reading level. Use short sentences and everyday words, and \
talk to the student directly ("you"). Avoid academic phrasing such as "we observe that", \
"it follows that", "hence", "thus", or "without loss of generality".
- Explain why each step works, not just what to compute, but keep it short. Skip routine \
arithmetic a middle school student does without thinking.
- If a quick check or a look at the answer choices makes the problem easier, mention it.
- Write math in LaTeX between $ signs, for example $\\frac{3}{4}$.
- End with a final line in exactly this form, using the letter of the correct choice:
Answer: (X)
If the problem has no answer choices, write the answer itself after "Answer:" instead."""

ALTERNATE_LLM_PROMPT = """\
You are an experienced math coach writing solutions to math competition problems for middle \
school students.

Guidelines:
- Write a correct solution that a strong middle school student could follow, and could \
have come up with during the contest without a calculator.
- Use only math that an advanced middle school student knows. Do not use trigonometry, \
complex numbers, calculus, or other high school or college math, even when it would be \
shorter.
- Write at an 8th to 10th grade reading level.
- You can skip steps that an honors Algebra 2 teacher would accept without explanation. If \
you skip several steps in a row, add a short sentence explaining what you did.
- Look for other solutions that use middle school competition math beyond the regular \
classroom curriculum. If you find more than one good solution, write them all up.
- Shortcut solutions are valuable, but always include at least one straightforward \
solution alongside any shortcut.
- If a quick check or a look at the answer choices makes the problem easier, mention it. For \
example, if you can work out the units digit of the answer, or that the answer is a multiple \
of 3, and that rules out some answer choices, include it in your solution.
- Write math in LaTeX between $ signs, for example $\\frac{3}{4}$.
- If you write more than one solution, label them "Solution 1", "Solution 2", and so on.
- End with a single final line in exactly this form, using the letter of the correct choice:
Answer: (X)
If the problem has no answer choices, write the answer itself after "Answer:" instead."""


def load_answer_keys(paths):
    """Answer key entries by problem ID, from files like local-only/aops-answer-keys/*.json."""
    keys = {}
    for path in paths or []:
        keys.update(json.loads(Path(path).read_text(encoding="utf-8"))["problems"])
    return keys


def build_messages(problem):
    user = problem.build_user_prompt(include_answer=False)
    if problem.answer_choice or problem.answer_text:
        choice = problem.answer_choice
        text = problem.answer_text
        answer = f"({choice}) {text}" if choice and text else (choice or text)
        user += (f"\n\nCorrect answer (from the official answer key): {answer}\n"
                 "Use it to check your work, but write the solution as if you worked it out "
                 "yourself, without mentioning that you were given the answer.")
    return [{"role": "system", "content": ALTERNATE_LLM_PROMPT},
            {"role": "user", "content": user}]


def message_text(message):
    """Plain text from an OpenRouter chat message (string or content blocks)."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""


def parse_answer(text):
    """The letter (or value) on the last "Answer:" line, ignoring Markdown bold."""
    matches = re.findall(r"^\W*Answer:\W*(.+?)\s*$", text, re.MULTILINE)
    if not matches:
        return None
    m = re.fullmatch(r"\(?\$?\\?\(?([A-E])\)?\$?\)?\**", matches[-1])
    return m.group(1) if m else matches[-1].strip("*").strip()


def solve(problem, provider, model, api_key, reasoning_effort=None):
    messages = build_messages(problem)
    body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": MAX_TOKENS}
    if reasoning_effort:
        if provider == "openrouter":
            body["reasoning"] = {"effort": reasoning_effort}
        else:
            body["reasoning_effort"] = reasoning_effort
    p = PROVIDERS[provider]
    resp = post_openrouter(body, api_key, timeout=p["timeout"], url=p["url"])
    choice = resp["choices"][0]
    solution = message_text(choice["message"])
    return {
        "problemId": problem.id,
        "knownAs": problem.known_as,
        "statement": problem.statement,
        "answerChoices": problem.prompt_answer_choices(),
        "model": model,
        "reasoningEffort": reasoning_effort,
        "answer": parse_answer(solution),
        "correctChoice": problem.answer_choice,
        "solution": solution,
        "finishReason": choice.get("finish_reason"),
        "usage": resp.get("usage"),
        "prompt": messages,
    }


def to_markdown(rec):
    lines = [f"## {rec['knownAs']} ({rec['problemId']})", "", rec["statement"].strip(), ""]
    lines += [f"({k}) {v}" for k, v in rec["answerChoices"].items()]
    lines += ["", "### Solution", "", rec["solution"], ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")
    ap.add_argument("problem_ids", nargs="*", metavar="PROBLEM_ID",
                    help="problem IDs, or @FILE to read them from a file with one ID per line")
    ap.add_argument("--sample", type=int, help="also solve N random problems without diagrams")
    ap.add_argument("--seed", type=int, help="random seed for --sample")
    ap.add_argument("--input", nargs="+", help="problem JSON file(s) (default: output/*.json)")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER,
                    help=f"which API to call (default {DEFAULT_PROVIDER})")
    ap.add_argument("--model", help=f"model ID (default {DEFAULT_MODEL} on openrouter, else the provider's default)")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"],
                    default=DEFAULT_REASONING_EFFORT,
                    help=f"reasoning depth on OpenRouter (default {DEFAULT_REASONING_EFFORT})")
    ap.add_argument("--out-dir", default="output/solutions", help="where to save results (default output/solutions)")
    ap.add_argument("--combined", metavar="PATH", help="also write all the solutions to one Markdown file")
    ap.add_argument("--force", action="store_true",
                    help="call the API even if a saved solution with the same prompt exists")
    ap.add_argument("--answer-key", nargs="+", metavar="PATH",
                    help="answer key JSON file(s); the correct answer is included in the prompt for problems they cover")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt without calling the API")
    ap.add_argument("--manifest", metavar="PATH",
                    help="JSON file of options to use; options given on the command line take precedence")
    args = ap.parse_args()
    if args.manifest:
        manifest_args = ap.parse_args(manifest_argv(args.manifest, ap))
        ap.set_defaults(**{k: v for k, v in vars(manifest_args).items() if v != ap.get_default(k)})
        args = ap.parse_args()
    provider = PROVIDERS[args.provider]
    if args.model is None:
        args.model = DEFAULT_MODEL if args.provider == DEFAULT_PROVIDER else provider["default_model"]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    problems = load_problems(args.input or sorted(Path("output").glob("*.json")))
    ids = list(args.problem_ids)
    if args.sample:
        pool = sorted(pid for pid, p in problems.items() if not p.has_diagram() and pid not in ids)
        ids += random.Random(args.seed).sample(pool, min(args.sample, len(pool)))
    if not ids:
        ap.error("give at least one PROBLEM_ID or use --sample")
    api_key = None if args.dry_run else provider["key_file"].read_text(encoding="utf-8").strip()
    out_dir = results_dir(args.out_dir, args.model, args.reasoning_effort)
    answer_keys = load_answer_keys(args.answer_key)
    combined = []

    for pid in ids:
        problem = problems.get(pid)
        if problem is None:
            print(f"{pid}: not found in input files", file=sys.stderr)
            continue
        if problem.has_diagram():
            print(f"{pid}: skipped (has a diagram)", file=sys.stderr)
            continue
        run_problem = problem.with_answer_key(answer_keys.get(pid))
        if args.dry_run:
            system, user = (m["content"] for m in build_messages(run_problem))
            print(f"\n=== {pid} ===\n--- system ---\n{system}\n--- user ---\n{user}")
            continue
        out_path = out_dir / f"{pid}.json"
        saved = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
        if (saved and saved.get("prompt") == build_messages(run_problem)
                and saved.get("reasoningEffort") == args.reasoning_effort and not args.force
                and saved.get("solution")):
            rec = saved
            print(f"\n(using saved solution {out_path})", file=sys.stderr)
        else:
            print(f"\nSolving {problem.known_as} ({pid})...", file=sys.stderr, flush=True)
            rec = solve(run_problem, args.provider, args.model, api_key, args.reasoning_effort)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n=== {rec['knownAs']} ({pid}) ===\n{rec['solution']}")
        if rec["finishReason"] != "stop":
            print(f"(finish reason: {rec['finishReason']})")
        combined.append(to_markdown(rec))

    if not args.dry_run:
        print(f"\nSaved solutions to {out_dir}", file=sys.stderr)
        if args.combined:
            path = Path(args.combined)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(combined), encoding="utf-8")
            print(f"Wrote {len(combined)} solutions to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
