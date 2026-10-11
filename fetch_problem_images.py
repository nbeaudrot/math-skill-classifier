"""Download problem diagram images referenced in fetch_problems.py JSON output.

Images are saved under output/images/<json-basename>/<problemId>/ so classify_skills.py
can attach them to vision-model prompts.

Examples:
    python fetch_problem_images.py output/amc8-2022.json
    python fetch_problem_images.py output/amc8-2022.json --out-dir output/images/amc8-2022
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


def image_refs(problem):
    """Yield (filename_stem, url) for every image on a problem."""
    d = problem.get("details", {})
    img = d.get("image")
    if img and img.get("url"):
        yield "statement", img["url"]
    for part in d.get("statementParts") or []:
        if part.get("kind") == "image" and part.get("url"):
            yield part.get("id") or "statement-part", part["url"]
    for letter in sorted((d.get("optionImages") or {}).keys()):
        opt = d["optionImages"][letter]
        if opt.get("url"):
            yield f"option-{letter}", opt["url"]


def download_url(url, dest, retries=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    req = urllib.request.Request(url, headers={"User-Agent": "math-skill-classifier/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return False


def extension_from_url(url):
    path = url.split("?", 1)[0]
    m = re.search(r"\.(png|jpe?g|gif|webp)$", path, re.I)
    return m.group(1).lower() if m else "png"


def fetch_images(records, out_dir):
    """Download images for all problems in exam record(s). Return manifest {problemId: {stem: path}}."""
    manifest = {}
    downloaded = skipped = 0
    for rec in records:
        for problem in rec.get("problems", []):
            pid = problem["id"]
            paths = {}
            for stem, url in image_refs(problem):
                ext = extension_from_url(url)
                dest = out_dir / pid / f"{stem}.{ext}"
                if download_url(url, dest):
                    downloaded += 1
                else:
                    skipped += 1
                paths[stem] = str(dest.relative_to(out_dir.parent)).replace("\\", "/")
            if paths:
                manifest[pid] = paths
    return manifest, downloaded, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_file", help="problem JSON from fetch_problems.py")
    ap.add_argument("--out-dir", help="image root (default: output/images/<json-basename>)")
    args = ap.parse_args()

    json_path = Path(args.json_file)
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        records = [records]
    out_dir = Path(args.out_dir) if args.out_dir else Path("output/images") / json_path.stem
    manifest, downloaded, skipped = fetch_images(records, out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(manifest)} problem(s) with images; {downloaded} downloaded, {skipped} already present",
          file=sys.stderr)
    print(f"Wrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
