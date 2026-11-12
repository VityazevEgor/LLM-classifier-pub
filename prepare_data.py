"""Сборка входов модели и эталона по сплитам. Из meta.json берутся только вид, имя, namespace и окружение ресурса."""
import csv, hashlib, json, sys, datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
P = {k: (ROOT / v).resolve() for k, v in CFG["paths"].items()}
ALLOWED_META = ["resource_kind", "resource_name", "namespace", "environment"]
FORBIDDEN = {"mutation_operators", "operator_params", "operator_split", "case_type", "pair_id", "app", "chart"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    splits = json.loads(P["splits"].read_text(encoding="utf-8"))
    dev, test = splits["dev"], splits["test"]
    assert len(dev) == 70 and len(test) == 252, (len(dev), len(test))
    assert not set(dev) & set(test), "dev/test пересекаются"
    with open(P["labels_main"], encoding="utf-8", newline="") as f:
        rows = {r["case_id"]: r for r in csv.DictReader(f)}
    assert len(rows) == 322, len(rows)
    assert set(rows) == set(dev) | set(test)
    (ROOT / "data" / "input").mkdir(parents=True, exist_ok=True)
    for cid, r in rows.items():
        assert r["label"] in {"1", "2", "3", "4"}, (cid, r["label"])
        d = P["raw"] / cid
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        rec = {"case_id": cid, **{k: meta[k] for k in ALLOWED_META}}
        rec["diff"] = json.loads((d / "diff.json").read_text(encoding="utf-8"))
        rec["diff_readable"] = r["diff_readable"]
        rec["desired_yaml"] = (d / "desired.yaml").read_text(encoding="utf-8")
        assert not FORBIDDEN & set(rec)
        (ROOT / "data" / "input" / f"{cid}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, ids in (("dev", dev), ("test", test)):
        with open(ROOT / "data" / f"labels_{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["case_id", "label"])
            for cid in sorted(ids): w.writerow([cid, rows[cid]["label"]])
    manifest = {"date": datetime.date.today().isoformat(),
                "sha256": {"annotation_main_labeled.csv": sha(P["labels_main"]),
                           "annotation_second_labeled.csv": sha(P["labels_second"])},
                "n_cases": 322, "n_dev": len(dev), "n_test": len(test)}
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("OK", manifest)


if __name__ == "__main__":
    sys.exit(main())
