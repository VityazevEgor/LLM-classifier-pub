"""python evaluate.py --model deepseek --split dev --prompt-version 1
   python evaluate.py --model deepseek --split test
index.csv используется только для срезов метрик на test."""
import argparse, csv, json, hashlib, datetime, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             cohen_kappa_score, confusion_matrix)
from llm_classifier.parse import vote

ROOT = Path(__file__).parent
CLASSES = [1, 2, 3, 4]


def macro_f1(y, p, only_present=False):
    """only_present=True (срезы): усреднение только по классам, присутствующим в эталоне."""
    labels = sorted(set(np.asarray(y).tolist()) & set(CLASSES)) if only_present else CLASSES
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def kappa(y, p, weights=None):
    v = cohen_kappa_score(y, p, labels=[0, 1, 2, 3, 4], weights=weights)
    return float(v) if np.isfinite(v) else 1.0


def fleiss_kappa(M):
    """M: (n_subjects, n_categories) счётчики оценок; одинаковое число оценщиков."""
    M = np.asarray(M, float)
    n = M.sum(1)[0]
    p_j = M.sum(0) / M.sum()
    P_i = ((M ** 2).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P_i.mean(), (p_j ** 2).sum()
    return 1.0 if Pe == 1 else float((Pbar - Pe) / (1 - Pe))


def metrics_block(y, p, only_present=False):
    y, p = np.asarray(y), np.asarray(p)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)), "macro_f1": macro_f1(y, p, only_present)}


def full_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    valid = p != 0
    return {
        **metrics_block(y, p),
        "per_class": {str(c): {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                               "support": int(sup[i])} for i, c in enumerate(CLASSES)},
        "confusion": {"rows_gold": CLASSES, "cols_pred": [0, 1, 2, 3, 4],
                      "matrix": confusion_matrix(y, p, labels=[0, 1, 2, 3, 4])[1:].tolist()},
        "kappa": kappa(y, p), "kappa_linear": kappa(y, p, "linear"),
        "under_rate": float(((p < y) & valid).mean()), "over_rate": float(((p > y) & valid).mean()),
        "invalid_count": int((p == 0).sum()),
    }


def bootstrap_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    acc, f1s, kw = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        acc.append(accuracy_score(y[i], p[i])); f1s.append(macro_f1(y[i], p[i]))
        kw.append(kappa(y[i], p[i], "linear"))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"accuracy": q(acc), "macro_f1": q(f1s), "kappa_linear": q(kw), "n_boot": n_boot, "seed": seed}


def load_runs(model, split, pv):
    suffix = f"_v{pv}" if split == "dev" else ""
    files = sorted(glob.glob(str(ROOT / "results" / f"raw_{model}_{split}{suffix}_run*.jsonl")),
                   key=lambda f: int(re.search(r"run(\d+)", f).group(1)))
    runs, recs = [], []
    for f in files:
        d = {}
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); d[r["case_id"]] = r
        runs.append(d); recs += list(d.values())
    return runs, recs


def rel(a, b):
    return "=" if a == b else ("<" if a < b else ">")


def slices_and_pairs(ids, gold, pred, runs_labels):
    """Срезы метрик на test по метаданным из index.csv."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    idx = pd.read_csv((ROOT / cfg["paths"]["index_csv"]).resolve()).set_index("case_id")
    print("operator_split values:", idx.loc[ids, "operator_split"].value_counts().to_dict())
    g, p = dict(zip(ids, gold)), dict(zip(ids, pred))
    groups = {
        "operator_split": lambda c: "seen" if idx.loc[c, "operator_split"] == "seen" else "held_out",
        "case_type": lambda c: idx.loc[c, "case_type"],
        "environment": lambda c: idx.loc[c, "environment"],
    }
    slices = {}
    for name, fn in groups.items():
        by = {}
        for c in ids:
            by.setdefault(fn(c), []).append(c)
        slices[name] = {k: metrics_block([g[c] for c in v], [p[c] for c in v], only_present=True) for k, v in sorted(by.items())}
    # согласованность пар
    pairs = {}
    for c in ids:
        if idx.loc[c, "case_type"] == "paired" and isinstance(idx.loc[c, "pair_id"], str):
            pairs.setdefault(idx.loc[c, "pair_id"], []).append(c)
    ok = tot = 0
    for pid, cs in pairs.items():
        envs = {idx.loc[c, "environment"]: c for c in cs}
        if len(cs) == 2 and set(envs) == {"dev", "prod"}:
            tot += 1
            ok += rel(p[envs["dev"]], p[envs["prod"]]) == rel(g[envs["dev"]], g[envs["prod"]])
    return slices, {"pairs_total": tot, "consistent": ok, "rate": ok / tot if tot else None}


def vs_expert2(ids, gold, pred, second_path):
    s = pd.read_csv(second_path).set_index("case_id")["label"]
    common = [c for c in ids if c in s.index]
    g = dict(zip(ids, gold)); p = dict(zip(ids, pred))
    e1 = [g[c] for c in common]; e2 = [int(s[c]) for c in common]; m = [p[c] for c in common]
    return {"n": len(common), "kappa_model_e2": kappa(e2, m), "kappa_linear_model_e2": kappa(e2, m, "linear"),
            "kappa_e1_e2": kappa(e2, e1), "kappa_linear_e1_e2": kappa(e2, e1, "linear"),
            "kappa_model_e1": kappa(e1, m), "accuracy_model_e2": float(np.mean(np.array(e2) == np.array(m)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--prompt-version", type=int)
    ap.add_argument("--prompt-sha-file", default=str(ROOT / "results" / "prompt_frozen.sha256"))
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pv = a.prompt_version or cfg["prompt_version"]
    lab = pd.read_csv(ROOT / "data" / f"labels_{a.split}.csv").sort_values("case_id")
    ids, gold = lab.case_id.tolist(), lab.label.tolist()
    runs, recs = load_runs(a.model, a.split, pv)
    assert runs, "нет файлов прогонов"
    for d in runs:
        missing = set(ids) - set(d)
        assert not missing, f"в прогоне нет {len(missing)} кейсов"
    run_labels = [[d[c]["label"] for c in ids] for d in runs]
    pred = [vote([rl[i] for rl in run_labels]) for i in range(len(ids))]
    out = {"model": a.model, "model_uri": recs[0]["model_uri"], "split": a.split,
           "run_date": sorted({r["timestamp"][:10] for r in recs}), "prompt_version": pv,
           "served_models": sorted({str(r.get("served_model")) for r in recs}), "n_runs": len(runs)}
    out.update(full_metrics(gold, pred))
    out["ci95"] = bootstrap_ci(gold, pred)
    out["prompt_sha256"] = (open(a.prompt_sha_file).read().strip() if Path(a.prompt_sha_file).exists() and a.split == "test" else None)
    out["tokens_total"] = {"prompt": sum(r["prompt_tokens"] for r in recs),
                           "completion": sum(r["completion_tokens"] for r in recs)}
    out["calls_stats"] = {"length_retry_cases": sum(bool(r["length_retry"]) for r in recs),
                          "invalid_retry_total": sum(r["invalid_retries"] for r in recs),
                          "truncated_yaml_cases": sum(bool(r["truncated"]) for r in runs[0].values())}
    per_run = [macro_f1(gold, rl) for rl in run_labels]
    stab = {"per_run_macro_f1": per_run, "per_run_accuracy": [float(accuracy_score(gold, rl)) for rl in run_labels],
            "macro_f1_min": min(per_run), "macro_f1_max": max(per_run),
            "per_run_invalid": [int(sum(l == 0 for l in rl)) for rl in run_labels]}
    if len(runs) >= 2:
        arr = np.array(run_labels).T
        stab["all_agree_share"] = float(np.mean([len(set(r)) == 1 for r in arr]))
        stab["fleiss_kappa"] = fleiss_kappa([[int((r == c).sum()) for c in range(5)] for r in arr])
    out["stability"] = stab
    if a.split == "test":
        out["slices"], out["pair_consistency"] = slices_and_pairs(ids, gold, pred, run_labels)
        out["vs_expert2"] = vs_expert2(ids, gold, pred, (ROOT / cfg["paths"]["labels_second"]).resolve())
    name = f"metrics_{a.model}_{a.split}" + (f"_v{pv}" if a.split == "dev" else "")
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.split == "test":
        with open(ROOT / "results" / f"predictions_{a.model}_test.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["case_id", "label", "run1", "run2", "run3", "explanation"])
            for i, c in enumerate(ids):
                rl = [r[i] for r in run_labels]
                src = next((j for j, l in enumerate(rl) if l == pred[i]), 0)
                w.writerow([c, pred[i], *rl, runs[src][c]["explanation"] or ""])
    print(json.dumps({k: out[k] for k in ("n", "accuracy", "macro_f1", "kappa_linear", "under_rate", "over_rate",
                                           "invalid_count", "tokens_total", "calls_stats")}, ensure_ascii=False))
    print("confusion (gold rows 1..4, pred cols 0..4):")
    for r in out["confusion"]["matrix"]:
        print(r)


if __name__ == "__main__":
    main()
