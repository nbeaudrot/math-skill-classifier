"""Problem and skill objects used across the classifier pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _dumps(data: Any, *, indent: int | None = None, **kwargs) -> str:
    opts = {"ensure_ascii": False, **kwargs}
    if indent is not None:
        opts["indent"] = indent
    return json.dumps(data, **opts)


IMPORTANCE_ORDER = {"essential": 0, "time-saving": 1, "helpful": 2}


@dataclass
class Skill:
    skill: str
    importance: str
    why: str
    category: str | None = None
    listed: bool | None = None
    new: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        return cls(
            skill=str(data.get("skill", "")),
            importance=str(data.get("importance", "")),
            why=str(data.get("why", "")),
            category=data.get("category"),
            listed=data.get("listed"),
            new=data.get("new"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "skill": self.skill,
            "importance": self.importance,
            "why": self.why,
        }
        if self.category is not None:
            out["category"] = self.category
        if self.listed is not None:
            out["listed"] = self.listed
        if self.new is not None:
            out["new"] = self.new
        return out

    def to_json(self, *, indent: int | None = None, **kwargs) -> str:
        return _dumps(self.to_dict(), indent=indent, **kwargs)


@dataclass
class ClassificationResult:
    summary: str
    overall_category: str
    skills: list[Skill] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationResult:
        return cls(
            summary=str(data.get("summary", "")),
            overall_category=str(data.get("overall_category", "")),
            skills=[Skill.from_dict(s) for s in data.get("skills") or []],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "overall_category": self.overall_category,
            "skills": [s.to_dict() for s in self.skills],
        }

    def to_json(self, *, indent: int | None = None, **kwargs) -> str:
        return _dumps(self.to_dict(), indent=indent, **kwargs)


@dataclass
class Problem:
    id: str
    statement: str
    answer_choices: dict[str, str]
    image_urls: dict[str, str] = field(default_factory=dict)
    skills: list[Skill] = field(default_factory=list)
    known_as: str | None = None
    exam_title: str | None = None
    exam_type: str | None = None
    label: str | None = None
    has_diagram_refs: bool = False

    @property
    def image_url(self) -> str | None:
        """Primary statement diagram URL, if any."""
        if "statement" in self.image_urls:
            return self.image_urls["statement"]
        for stem, url in self.image_urls.items():
            if stem == "statement" or stem.startswith("statement"):
                return url
        return next(iter(self.image_urls.values()), None)

    def prompt_answer_choices(self) -> dict[str, str]:
        """Answer choices wrapped in $ for prompts and combined exports."""
        return {k: f"${self.answer_choices[k]}$" for k in sorted(self.answer_choices)}

    def has_diagram(self) -> bool:
        return self.has_diagram_refs or bool(self.image_urls)

    def local_image_paths(self, images_dir) -> list[Path]:
        """Downloaded images for this problem, in prompt order (statement, then A–E)."""
        if not images_dir:
            return []
        root = Path(images_dir) / self.id
        if not root.is_dir():
            return []
        ordered: list[Path] = []
        for stem in ("statement",):
            for path in sorted(root.glob(f"{stem}.*")):
                ordered.append(path)
                break
        for letter in "ABCDE":
            for path in sorted(root.glob(f"option-{letter}.*")):
                ordered.append(path)
                break
        return ordered

    def build_user_prompt(self, answer_entry: dict[str, Any] | None = None) -> str:
        lines = [f"Source: {self.known_as or self.exam_title}"]
        if self.exam_type:
            lines.append(f"Competition level: {self.exam_type}")
        lines += ["", "Problem:", self.statement]
        choices = self.prompt_answer_choices()
        if choices:
            lines += ["", "Answer choices:"]
            lines += [f"({k}) {v}" for k, v in choices.items()]
        if answer_entry:
            lines += ["", "Known correct answer (from the official answer key):"]
            if answer_entry.get("correctChoice"):
                lines.append(f"Choice: ({answer_entry['correctChoice']})")
            if answer_entry.get("answerText"):
                lines.append(f"Answer: {answer_entry['answerText']}")
        return "\n".join(lines)

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        *,
        exam_title: str | None = None,
        exam_type: str | None = None,
    ) -> Problem:
        details = record.get("details") or {}
        opts = details.get("options") or {}
        return cls(
            id=record["id"],
            statement=(details.get("statement") or "").strip(),
            answer_choices={k: opts[k] for k in sorted(opts)},
            image_urls=dict(_image_url_refs(record)),
            known_as=record.get("knownAs"),
            exam_title=exam_title,
            exam_type=exam_type,
            label=record.get("label"),
            has_diagram_refs=has_diagram_in_record(record),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "statement": self.statement,
            "answer_choices": self.answer_choices,
            "image_urls": self.image_urls,
            "skills": [s.to_dict() for s in self.skills],
            "has_diagram_refs": self.has_diagram_refs,
        }
        if self.known_as is not None:
            out["known_as"] = self.known_as
        if self.exam_title is not None:
            out["exam_title"] = self.exam_title
        if self.exam_type is not None:
            out["exam_type"] = self.exam_type
        if self.label is not None:
            out["label"] = self.label
        return out

    def to_json(self, *, indent: int | None = None, **kwargs) -> str:
        return _dumps(self.to_dict(), indent=indent, **kwargs)


def _image_url_refs(problem: dict[str, Any]) -> list[tuple[str, str]]:
    """(filename_stem, url) pairs, same order as fetch_problem_images.image_refs."""
    d = problem.get("details") or {}
    refs: list[tuple[str, str]] = []
    img = d.get("image")
    if img and img.get("url"):
        refs.append(("statement", img["url"]))
    for part in d.get("statementParts") or []:
        if part.get("kind") == "image" and part.get("url"):
            refs.append((part.get("id") or "statement-part", part["url"]))
    for letter in sorted((d.get("optionImages") or {}).keys()):
        opt = d["optionImages"][letter]
        if opt.get("url"):
            refs.append((f"option-{letter}", opt["url"]))
    return refs


def has_diagram_in_record(problem: dict[str, Any]) -> bool:
    """True when the API record references diagram images (even if not downloaded)."""
    d = problem.get("details") or {}
    if d.get("image") or d.get("optionImages"):
        return True
    return any(part.get("kind") == "image" for part in d.get("statementParts") or [])


def load_problems(paths) -> dict[str, Problem]:
    import json
    from pathlib import Path

    problems: dict[str, Problem] = {}
    for path in paths:
        for rec in json.loads(Path(path).read_text(encoding="utf-8")):
            exam_title = rec["title"]
            exam_type = rec["exam"].get("examType")
            for p in rec["problems"]:
                prob = Problem.from_record(p, exam_title=exam_title, exam_type=exam_type)
                problems[prob.id] = prob
    return problems


def match_skills(result: ClassificationResult | dict[str, Any], skills_by_name: dict) -> ClassificationResult:
    """Normalize skill names against the master list and sort by importance."""
    if isinstance(result, dict):
        result = ClassificationResult.from_dict(result)
    ordered = sorted(
        result.skills,
        key=lambda s: IMPORTANCE_ORDER.get(str(s.importance).lower(), 3),
    )
    matched: list[Skill] = []
    for s in ordered:
        known = skills_by_name.get(s.skill.strip().lower())
        if known:
            matched.append(
                Skill(
                    skill=known["skill"],
                    importance=s.importance,
                    why=s.why,
                    category=known["category"],
                    listed=True,
                )
            )
        else:
            matched.append(
                Skill(
                    skill=s.skill,
                    importance=s.importance,
                    why=s.why,
                    category=s.category,
                    listed=False,
                    new=s.new if s.new is not None else True,
                )
            )
    return ClassificationResult(summary=result.summary, overall_category=result.overall_category, skills=matched)
