import hashlib
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
TRUNC_NOTE = "\n[... обрезано ...]"


def load_prompt(version, directory=None):
    d = Path(directory) if directory else PROMPTS
    return ((d / f"prompt_v{version}.md").read_text(encoding="utf-8"),
            (d / f"user_template_v{version}.md").read_text(encoding="utf-8"))


def prompt_sha256(system, template):
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    return {"system": h(system), "user_template": h(template),
            "combined": h(h(system) + h(template))}


def build_user_message(case, template, limit=20000):
    y = case["desired_yaml"]
    truncated = len(y) > limit
    if truncated:
        y = y[:limit] + TRUNC_NOTE
    msg = template.format(resource_kind=case["resource_kind"], resource_name=case["resource_name"],
                          namespace=case["namespace"], environment=case["environment"],
                          diff_readable=case["diff_readable"], desired_yaml=y)
    return msg, truncated
