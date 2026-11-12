"""python run.py --model deepseek|qwen --split dev|test --runs N [--prompt-version V] [--prompt-dir DIR]
   python run.py --model deepseek|qwen --check [CASE_ID]   # быстрая проверка: один кейс, вывод в консоль, без записи файлов"""
import argparse, json, threading, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from llm_classifier.client import YandexClient, load_env
from llm_classifier.parse import parse_response
from llm_classifier.prompt import load_prompt, build_user_message

ROOT = Path(__file__).parent


def done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["case_id"], r["run"]))
    return keys


def classify(client, uri, system, user, cfg):
    """Невалидный ответ -> до invalid_retries повторов; пустой content + length -> повтор с max_tokens_retry."""
    rec = {"invalid_retries": 0, "length_retry": False, "prompt_tokens": 0, "completion_tokens": 0}
    latency = 0.0
    for attempt in range(1 + cfg["invalid_retries"]):
        resp = client.chat(uri, system, user, cfg["max_tokens"])
        for _ in range(2):
            ch = resp["choices"][0]
            msg = ch["message"]
            latency += resp["_latency"]
            for k in ("prompt_tokens", "completion_tokens"):
                rec[k] += (resp.get("usage") or {}).get(k, 0)
            if not msg.get("content") and ch.get("finish_reason") == "length" and not rec["length_retry"]:
                rec["length_retry"] = True
                resp = client.chat(uri, system, user, cfg["max_tokens_retry"])
                continue
            break
        label, expl = parse_response(msg.get("content"))
        rec.update(content=msg.get("content"), reasoning_content=msg.get("reasoning_content"),
                   finish_reason=ch.get("finish_reason"), label=label, explanation=expl,
                   served_model=resp.get("model"), invalid_retries=attempt)
        if label != 0:
            break
    rec["latency_s"] = round(latency, 2)
    return rec


def check(client, uri, system, template, cfg, case_id):
    """Один запрос по одному кейсу: печатает ответ модели и эталон, ничего не записывает на диск."""
    gold = {}
    for split in ("test", "dev"):
        for line in (ROOT / "data" / f"labels_{split}.csv").read_text(encoding="utf-8").splitlines()[1:]:
            cid, lab = line.split(",")
            gold[cid] = int(lab)
    case_id = case_id or sorted(c for c in gold if (ROOT / "data" / "input" / f"{c}.json").exists())[0]
    case = json.loads((ROOT / "data" / "input" / f"{case_id}.json").read_text(encoding="utf-8"))
    user, _ = build_user_message(case, template, cfg["call"]["desired_yaml_limit"])
    rec = classify(client, uri, system, user, cfg["call"])
    print(f"кейс: {case_id} ({case['resource_kind']} {case['resource_name']}, {case['environment']})")
    print("diff:", case["diff_readable"], sep="\n")
    print(f"модель: {uri.split('/', 3)[-1]}")
    print(f"предсказание: {rec['label']}; эталон: {gold[case_id]}; {'СОВПАДАЕТ' if rec['label'] == gold[case_id] else 'НЕ СОВПАДАЕТ'}")
    print(f"объяснение: {rec['explanation']}")
    print(f"токены: prompt {rec['prompt_tokens']}, completion {rec['completion_tokens']}; время {rec['latency_s']} с")
    return 0 if rec["label"] == gold[case_id] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--check", nargs="?", const="", metavar="CASE_ID",
                    help="один запрос по одному кейсу (по умолчанию первый по номеру); результат сравнивается с эталоном и печатается, файлы не пишутся")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--prompt-version", type=int)
    ap.add_argument("--prompt-dir")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pv = a.prompt_version or cfg["prompt_version"]
    system, template = load_prompt(pv, a.prompt_dir)
    env = load_env(str((ROOT / cfg["env_file"]).resolve()))
    client = YandexClient(env, cfg["call"])
    uri = client.model_uri(cfg["models"][a.model]["uri_template"])
    if a.check is not None:
        return check(client, uri, system, template, cfg, a.check)
    if not a.split:
        ap.error("укажите --split или --check")
    ids = sorted(l.split(",")[0] for l in
                 (ROOT / "data" / f"labels_{a.split}.csv").read_text(encoding="utf-8").splitlines()[1:])
    lock = threading.Lock()
    (ROOT / "results").mkdir(exist_ok=True)
    for k in range(1, a.runs + 1):
        suffix = f"_v{pv}" if a.split == "dev" else ""
        path = ROOT / "results" / f"raw_{a.model}_{a.split}{suffix}_run{k}.jsonl"
        done = done_keys(path)
        todo = [c for c in ids if (c, k) not in done]
        print(f"run{k}: {len(todo)} к запросу, {len(done)} уже есть", flush=True)

        def work(cid):
            case = json.loads((ROOT / "data" / "input" / f"{cid}.json").read_text(encoding="utf-8"))
            user, trunc = build_user_message(case, template, cfg["call"]["desired_yaml_limit"])
            try:
                rec = classify(client, uri, system, user, cfg["call"])
            except Exception as e:  # отказ после всех попыток: не пишем, при повторном запуске докачается
                print("FAIL", cid, str(e)[:200], flush=True)
                return
            rec.update(case_id=cid, run=k, prompt_version=pv, model_uri=uri,
                       timestamp=datetime.datetime.now().isoformat(timespec="seconds"), truncated=trunc)
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(cfg["call"]["max_workers"]) as ex:
            list(ex.map(work, todo))
        print(f"run{k}: готово, записей {len(done_keys(path))}", flush=True)


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
