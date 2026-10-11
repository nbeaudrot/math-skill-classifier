"""Ask an LLM which mathematical skills are needed to solve a problem.

Problems are looked up by ID in the JSON files written by fetch_problems.py.
When diagram images are available (see fetch_problem_images.py), they are attached to the prompt.

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

    # Read all options from a manifest file; options on the command line override it
    python classify_skills.py --manifest run-manifests/amc8-25-mgb.json
    #   {"problems": "sample-problem-lists/amc8-25questions.txt",
    #    "input": ["output/amc-8-2011-2025.json"], "provider": "mgb", "combined": "output/skills/amc8-25-mgb.json"}

    # Show the prompt without calling the API
    python classify_skills.py X4hTWCLinR4Gs8QLicVw --dry-run

    # Include the official answer to consider alternate solution paths
    python classify_skills.py @sample-problem-lists/amc8-2022-no-diagram.txt --input output/amc8-2022.json --answer-key

    # Attach diagrams downloaded from VentiMath (see fetch_problem_images.py)
    python fetch_problem_images.py output/amc8-2022.json
    python classify_skills.py nBHNiRFOP0h4LI6KDCfF --input output/amc8-2022.json --dry-run
"""

import argparse
import base64
import csv
import json
import mimetypes
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from models import Problem, Skill, load_problems, match_skills as _match_skills

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5@high"
# OpenRouter reserves credit for max_tokens up front; without a cap it uses the model's
# maximum (65k+), which a low-balance key can't cover. This includes reasoning tokens.
MAX_TOKENS = 16000
KEY_FILE = Path(__file__).parent / "local-only" / "openrouter_api_key.txt"
SKILLS_FILE = Path(__file__).parent / "skill-lists" / "skills-merged.csv"
DEFAULT_ANSWER_KEY = Path(__file__).parent / "local-only" / "aops-answer-keys" / "2022_amc_8_answer_key.json"
ANSWER_KEY_RUN_SUFFIX = "-with-key"

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

IMAGE_GUIDANCE = """\
Diagram images may be attached to the problem. Examining them may help you understand the \
problem and classify skills accurately, but sometimes an image is only an illustration to help \
students visualize the setting and is not required to solve the problem. Use the images when \
they carry mathematical information relevant to a reasonable solution."""

ANSWER_KEY_GUIDANCE = """\
Additional guidelines (the correct answer is given below):
- Use the known answer to sanity-check which skills fit a complete solution, but do not \
reverse-engineer a full solution in the response.
- Consider alternate reasonable solution paths, including clever formulas or problem-solving \
strategies common on math team and competition curricula (e.g. telescoping, complementary \
counting, Simon's Favorite Factoring Trick, divisibility shortcuts), and tag skills for any \
path that a strong student might actually use.
- If the answer makes a shortcut or test-taking approach clearly viable, include those skills \
when appropriate."""


def default_images_dir(input_paths):
    """output/images/<basename> when classifying from a single fetch_problems JSON file."""
    if not input_paths or len(input_paths) != 1:
        return None
    path = Path(input_paths[0])
    candidate = Path("output/images") / path.stem
    return candidate if candidate.is_dir() else None


def file_to_data_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_caption(path):
    name = path.stem
    if name == "statement":
        return "Problem diagram:"
    if name.startswith("option-"):
        return f"Answer choice ({name.rpartition('-')[2]}) diagram:"
    return f"Diagram ({name}):"





def load_answer_key(path):
    """Map problem ID -> {correctChoice, answerText, ...} from an AoPS-style answer key JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("problems") or {}


def system_prompt(include_answer_key=False, include_images=False):
    text = LLM_PROMPT
    if include_images:
        text += "\n\n" + IMAGE_GUIDANCE
    if include_answer_key:
        text += "\n\n" + ANSWER_KEY_GUIDANCE
    return text


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


REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})


def split_model_spec(model, reasoning_effort=None):
    """OpenRouter model IDs may be written model@high in model lists; split into API id and effort."""
    if reasoning_effort is not None:
        return model, reasoning_effort
    if "@" in model:
        base, _, suffix = model.rpartition("@")
        if suffix in REASONING_EFFORTS:
            return base, suffix
    return model, None


def run_label(model, reasoning_effort=None):
    """How a model + effort setting is named in folders and comparisons, e.g. google/gemini-3.8-flash@medium."""
    return f"{model}@{reasoning_effort}" if reasoning_effort else model


def parse_run_label(label):
    """Undo run_label; only treats @suffix as effort when suffix is a known reasoning level."""
    if "@" not in label:
        return label, None
    base, _, suffix = label.rpartition("@")
    if suffix in REASONING_EFFORTS:
        return base, suffix
    return label, None


def results_dir(out_dir, model, reasoning_effort=None, with_answer_key=False):
    label = run_label(model, reasoning_effort)
    if with_answer_key:
        label += ANSWER_KEY_RUN_SUFFIX
    return Path(out_dir) / re.sub(r"[^A-Za-z0-9.@-]+", "_", label)


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


def build_user_content(problem: Problem, answer_entry=None, images_dir=None):
    text = problem.build_user_prompt(answer_entry)
    paths = problem.local_image_paths(images_dir)
    if not paths:
        return text
    blocks = [{"type": "text", "text": text}]
    for path in paths:
        blocks.append({"type": "text", "text": image_caption(path)})
        blocks.append({"type": "image_url", "image_url": {"url": file_to_data_url(path)}})
    return blocks


def build_messages(problem: Problem, answer_entry=None, images_dir=None):
    include_key = answer_entry is not None
    paths = problem.local_image_paths(images_dir)
    include_images = bool(paths)
    user = build_user_content(problem, answer_entry, images_dir)
    return [{"role": "system", "content": system_prompt(include_key, include_images)},
            {"role": "user", "content": user}]


def match_skills(result):
    """Use the skill list's exact name and category, mark anything not on the list as unlisted,
    set new: true when the model omitted it, and order skills essential first (keeping the
    model's order within each group)."""
    if result is None:
        return None
    matched = _match_skills(result, SKILLS_BY_NAME)
    data = matched.to_dict()
    if isinstance(result, dict):
        result.clear()
        result.update(data)
        return result
    return matched


def classify(problem: Problem, model, api_key, reasoning_effort=None, provider="openrouter",
             answer_entry=None, images_dir=None):
    messages = build_messages(problem, answer_entry, images_dir)
    resp = call_llm(model, messages, api_key, reasoning_effort, provider)
    choice = resp["choices"][0]
    reply = choice["message"].get("content") or ""
    try:
        result = match_skills(parse_json_reply(reply))
    except (json.JSONDecodeError, AttributeError):
        result = None
    return {
        "problemId": problem.id,
        "knownAs": problem.known_as,
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
    skills = res.get("skills", [])
    for raw in skills:
        s = raw if isinstance(raw, dict) else raw.to_dict()
        flag = "" if s.get("listed") else "  ** NOT ON SKILL LIST **"
        print(f"  [{s.get('importance', '?'):11}] {s.get('skill')}  ({s.get('category')}){flag}")
        print(f"                {s.get('why')}")


def manifest_argv(path, parser):
    """Turn a manifest into command-line arguments, so argparse checks its values like any others.

    Keys are option names without the leading dashes ("reasoning-effort" or "reasoning_effort").
    "problems" is a list of problem IDs, or the path of a file with one ID per line."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    options = {a.dest for a in parser._actions if a.option_strings} - {"help", "manifest"}
    argv = []
    problems = manifest.pop("problems", [])
    argv += ["@" + problems] if isinstance(problems, str) else [str(p) for p in problems]
    for key, value in manifest.items():
        dest = key.replace("-", "_")
        if dest not in options:
            parser.error(f"{path}: unknown option {key!r}")
        flag = "--" + dest.replace("_", "-")
        if isinstance(value, bool):
            argv += [flag] if value else []
        elif isinstance(value, list):
            argv += [flag, *map(str, value)]
        elif value is not None:
            argv += [flag, str(value)]
    return argv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")
    ap.add_argument("problem_ids", nargs="*", metavar="PROBLEM_ID",
                    help="problem IDs, or @FILE to read them from a file with one ID per line")
    ap.add_argument("--sample", type=int, help="also classify N random problems without diagrams")
    ap.add_argument("--seed", type=int, help="random seed for --sample")
    ap.add_argument("--input", nargs="+", help="problem JSON file(s) (default: output/*.json)")
    ap.add_argument("--images-dir", metavar="PATH",
                    help="folder of images from fetch_problem_images.py "
                         "(default: output/images/<basename> when --input is one file)")
    ap.add_argument("--skip-diagrams-without-images", action="store_true",
                    help="skip problems that have VentiMath images but none on disk in --images-dir")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="openrouter",
                    help="which API to call (default openrouter; mgb is a friend's self-hosted Qwen server)")
    ap.add_argument("--model", help="model ID (default: provider default); "
                    "may include @low|medium|high|xhigh for reasoning effort")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"],
                    help="ask the model to think before answering (default: the model's own default)")
    ap.add_argument("--out-dir", default="output/skills", help="where to save results (default output/skills)")
    ap.add_argument("--combined", metavar="PATH",
                    help="also write one JSON array with each problem, its answer choices, and the model's result")
    ap.add_argument("--force", action="store_true",
                    help="call the API even if a saved result with the same prompt exists")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt without calling the API")
    ap.add_argument("--answer-key", metavar="PATH", nargs="?", const=str(DEFAULT_ANSWER_KEY),
                    help="include the official answer (choice and value) in the prompt; "
                         f"default file: {DEFAULT_ANSWER_KEY.name}")
    ap.add_argument("--manifest", metavar="PATH",
                    help="JSON file of options to use; options given on the command line take precedence")
    args = ap.parse_args()
    if args.manifest:
        manifest_args = ap.parse_args(manifest_argv(args.manifest, ap))
        ap.set_defaults(**{k: v for k, v in vars(manifest_args).items() if v != ap.get_default(k)})
        args = ap.parse_args()
    provider = PROVIDERS[args.provider]
    args.model = args.model or provider["default_model"]
    args.model, parsed_effort = split_model_spec(args.model, args.reasoning_effort)
    if parsed_effort is not None and args.reasoning_effort is None:
        args.reasoning_effort = parsed_effort
    # The Windows console defaults to cp1252, which can't print characters like the square root sign.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    input_paths = args.input or [str(p) for p in sorted(Path("output").glob("*.json"))]
    problems = load_problems(input_paths)
    images_dir = args.images_dir or default_images_dir(input_paths)
    ids = list(args.problem_ids)
    if args.sample:
        pool = sorted(pid for pid in problems if pid not in ids)
        ids += random.Random(args.seed).sample(pool, min(args.sample, len(pool)))
    if not ids:
        ap.error("give at least one PROBLEM_ID or use --sample")
    api_key = None if args.dry_run else provider["key_file"].read_text(encoding="utf-8").strip()
    answer_keys = load_answer_key(args.answer_key) if args.answer_key else None
    out_dir = results_dir(args.out_dir, args.model, args.reasoning_effort, bool(answer_keys))
    combined = []

    for pid in ids:
        problem = problems.get(pid)
        if problem is None:
            print(f"{pid}: not found in input files", file=sys.stderr)
            continue
        if args.skip_diagrams_without_images and problem.has_diagram() and not problem.local_image_paths(images_dir):
            print(f"{pid}: skipped (diagram but no local images; run fetch_problem_images.py)", file=sys.stderr)
            continue
        key_entry = answer_keys.get(problem.id) if answer_keys else None
        if answer_keys and key_entry is None:
            print(f"{pid}: warning: no answer key entry", file=sys.stderr)
        messages = build_messages(problem, key_entry, images_dir)
        if args.dry_run:
            user = messages[1]["content"]
            if isinstance(user, list):
                user_lines = []
                for block in user:
                    if block.get("type") == "text":
                        user_lines.append(block["text"])
                    elif block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        user_lines.append(f"[image: {len(url)} chars data URL]")
                user_display = "\n".join(user_lines)
            else:
                user_display = user
            img_note = ""
            img_paths = problem.local_image_paths(images_dir)
            if img_paths:
                img_note = f" ({len(img_paths)} image(s) from {images_dir})"
            print(f"\n=== {pid}{img_note} ===\n--- system ---\n{messages[0]['content']}\n--- user ---\n{user_display}")
            continue
        out_path = out_dir / f"{pid}.json"
        saved = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
        if (saved and saved.get("prompt") == messages
                and saved.get("reasoningEffort") == args.reasoning_effort and not args.force
                and reparse_saved(saved)["result"] is not None):
            rec = saved
            match_skills(rec["result"])
            print(f"\n(using saved result {out_path})", file=sys.stderr)
        else:
            rec = classify(problem, args.model, api_key, args.reasoning_effort, args.provider,
                           key_entry, images_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        if rec["result"] is not None:
            problem.skills = [Skill.from_dict(s) for s in rec["result"].get("skills", [])]
        print_result(rec)
        combined.append({
            "problemId": pid,
            "knownAs": problem.known_as,
            "statement": problem.statement,
            "answerChoices": problem.prompt_answer_choices(),
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
