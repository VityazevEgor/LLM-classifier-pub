import json
from pathlib import Path
import pytest
from llm_classifier.prompt import load_prompt, build_user_message, TRUNC_NOTE

ROOT = Path(__file__).resolve().parent.parent
# нужны: python prepare_data.py и python prompts/build_prompt.py 2
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "input").exists() or not (ROOT / "prompts" / "prompt_v2.md").exists(),
                                reason="сначала выполните prepare_data.py и prompts/build_prompt.py 2")
FORBIDDEN = ["mutation_operators", "operator_params", "operator_split", "case_type", "pair_id"]


def case():
    return json.loads(next((ROOT / "data" / "input").glob("CASE_*.json")).read_text(encoding="utf-8"))


def test_input_has_no_forbidden_keys():
    for f in (ROOT / "data" / "input").glob("CASE_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        assert not set(d) & {*FORBIDDEN, "app", "chart"}
        assert set(d) == {"case_id", "resource_kind", "resource_name", "namespace", "environment",
                          "diff", "diff_readable", "desired_yaml"}


def test_system_prompt_contains_rubric():
    s, _ = load_prompt(2)
    for frag in ["## 1. Что такое кейс", "## 3. Шкала классов", "### 4.3 Составные кейсы",
                 "## 6. Калибровочные примеры", "В1. Возникает ли прямой путь"]:
        assert frag in s
    assert "## 7. Порядок работы" not in s and "## 2. Что вы видите" not in s


def test_user_message_fields_and_no_leak():
    _, t = load_prompt(2)
    c = case()
    m, trunc = build_user_message(c, t)
    assert not trunc
    for v in (c["resource_kind"], c["resource_name"], c["namespace"], c["diff_readable"], c["desired_yaml"]):
        assert v in m
    for k in FORBIDDEN:
        assert k not in m


def test_truncation():
    _, t = load_prompt(2)
    c = case(); c["desired_yaml"] = "x" * 30000
    m, trunc = build_user_message(c, t, 20000)
    assert trunc and TRUNC_NOTE.strip() in m and "x" * 20001 not in m
