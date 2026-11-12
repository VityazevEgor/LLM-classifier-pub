import json
from collections import Counter

INVALID = 0


def parse_response(content):
    """Возвращает (label, explanation). Берётся первый JSON-объект с ключом label (иначе первый объект).
    Невалидно -> (0, None)."""
    if not content or not isinstance(content, str):
        return INVALID, None
    dec = json.JSONDecoder()
    objs, i = [], content.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(content, i)
            if isinstance(obj, dict):
                objs.append(obj)
                i = content.find("{", end)
                continue
        except ValueError:
            pass
        i = content.find("{", i + 1)
    if not objs:
        return INVALID, None
    obj = next((o for o in objs if "label" in o), objs[0])
    label, expl = obj.get("label"), obj.get("explanation")
    if isinstance(label, bool) or not isinstance(label, int) or label not in (1, 2, 3, 4):
        return INVALID, None
    if not isinstance(expl, str) or not expl.strip():
        return INVALID, None
    return label, expl.strip()


def vote(labels):
    """Мода по прогонам; если нет единственной моды (напр. все три различны) — ответ первого прогона."""
    c = Counter(labels)
    n = c.most_common(1)[0][1]
    tops = [k for k, v in c.items() if v == n]
    return tops[0] if len(tops) == 1 else labels[0]
