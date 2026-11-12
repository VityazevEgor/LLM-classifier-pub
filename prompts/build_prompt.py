"""Собирает системную часть промпта: роль + задача + разделы 1, 3, 4 (4.1-4.3), 6 инструкции разметки
(дословно из Rubric_DriftBench_Annotation.md репозитория DriftBench-Offline-Pub; путь — paths.rubric в config.yaml) + формат ответа. Использование: python build_prompt.py 1"""
import re, sys
from pathlib import Path
import yaml

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE.parent / "config.yaml").read_text(encoding="utf-8"))
rubric = (HERE.parent / CFG["paths"]["rubric"]).resolve().read_text(encoding="utf-8")


def section(n: int) -> str:
    m = re.search(rf"^## {n}\. .*?(?=^## \d+\. |\Z)", rubric, re.S | re.M)
    return m.group(0).strip()


def build(version: int) -> str:
    intro = open(HERE / f"_intro_v{version}.md", encoding="utf-8").read().strip()
    outro = open(HERE / f"_outro_v{version}.md", encoding="utf-8").read().strip()
    body = "\n\n".join(section(n) for n in (1, 3, 4, 6))
    return f"{intro}\n\n# ИНСТРУКЦИЯ ПО РАЗМЕТКЕ\n\n{body}\n\n{outro}\n"


if __name__ == "__main__":
    v = int(sys.argv[1])
    (HERE / f"prompt_v{v}.md").write_text(build(v), encoding="utf-8", newline="\n")
    print("written", v)
