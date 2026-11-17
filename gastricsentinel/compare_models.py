"""Statistical comparison of two trained image models on the SAME held-out test images.

    python -m gastricsentinel.compare_models <A> <B> [<C> ...]      # A is compared with every other
    A, B ... are version ids (artifacts/versions/<id>) or folders containing test_predictions.csv

Reports, for each pair: accuracy and macro-F1 of both, an exact McNemar test on the paired correct/incorrect
outcomes, and 95% paired-bootstrap intervals for the accuracy and macro-F1 differences. Results go to
artifacts/comparisons/<A>__<B>.json, which the web app's Models page reads.

Read it honestly: the test is image-level. With a patch-level split, images from one patient are correlated,
so p-values are optimistic. Many pairwise tests also need a multiple-comparison correction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from .common import load_json, save_json
from .config import ARTIFACTS


def _resolve(x, art):
    p = Path(x)
    if (p / "test_predictions.csv").exists():
        return p
    p = Path(art) / "versions" / x
    if (p / "test_predictions.csv").exists():
        return p
    raise SystemExit(f"cannot find test_predictions.csv for '{x}'")


def _label(d: Path) -> dict:
    m = load_json(d / "meta.json") if (d / "meta.json").exists() else {}
    return {"id": m.get("id", d.name), "arch": m.get("arch"), "note": m.get("note", "")}


def compare(dir_a, dir_b, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = pd.read_csv(Path(dir_a) / "test_predictions.csv"), pd.read_csv(Path(dir_b) / "test_predictions.csv")
    j = a.merge(b, on="path", suffixes=("_a", "_b"))
    out = {"a": _label(Path(dir_a)), "b": _label(Path(dir_b)), "n_a": len(a), "n_b": len(b), "n_paired": len(j), "warnings": []}
    if len(j) < 10:
        raise SystemExit("fewer than 10 shared test images; the two models were not tested on the same split")
    if len(j) < min(len(a), len(b)):
        out["warnings"].append(f"Only {len(j)} of {min(len(a), len(b))} test images are shared; compare models trained with the same data and seed.")
    if (j["true_a"] != j["true_b"]).any():
        raise SystemExit("labels disagree for shared images; the two runs used different data")
    y = j["true_a"].to_numpy()
    pa, pb = j["pred_a"].to_numpy(), j["pred_b"].to_numpy()
    ok_a, ok_b = pa == y, pb == y
    n01, n10 = int((~ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())     # only B right / only A right
    out["mcnemar"] = {"only_a_correct": n10, "only_b_correct": n01,
                      "p_value": float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0,
                      "test": "exact two-sided McNemar (binomial on discordant pairs)"}
    labels = sorted(set(y))
    f = lambda t, p: f1_score(t, p, labels=labels, average="macro", zero_division=0)  # noqa: E731
    out["a"].update(accuracy=float(ok_a.mean()), macro_f1=float(f(y, pa)))
    out["b"].update(accuracy=float(ok_b.mean()), macro_f1=float(f(y, pb)))
    rng = np.random.default_rng(seed)
    da, df_ = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        da.append(ok_b[i].mean() - ok_a[i].mean())
        df_.append(f(y[i], pb[i]) - f(y[i], pa[i]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out["diff_b_minus_a"] = {"accuracy": float(ok_b.mean() - ok_a.mean()), "accuracy_ci95": ci(da),
                             "macro_f1": float(f(y, pb) - f(y, pa)), "macro_f1_ci95": ci(df_), "n_boot": n_boot}
    sig = out["mcnemar"]["p_value"] < 0.05
    out["verdict"] = ("A statistically detectable difference in accuracy (p < 0.05, before any multiple-comparison correction)."
                      if sig else "No statistically detectable difference in accuracy on this test set.")
    out["caveat"] = "Image-level test; optimistic if images share patients. Correct for multiple comparisons when testing many models."
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if len(a.models) < 2:
        raise SystemExit("give at least two models")
    dirs = [_resolve(m, a.artifacts) for m in a.models]
    dest = Path(a.artifacts) / "comparisons"
    dest.mkdir(parents=True, exist_ok=True)
    for d in dirs[1:]:
        r = compare(dirs[0], d, a.n_boot)
        save_json(r, dest / f"{r['a']['id']}__{r['b']['id']}.json")
        dd = r["diff_b_minus_a"]
        print(f"{r['a']['id']} ({r['a']['arch']}) vs {r['b']['id']} ({r['b']['arch']}): n={r['n_paired']}  "
              f"acc {r['a']['accuracy']:.3f} vs {r['b']['accuracy']:.3f}  McNemar p={r['mcnemar']['p_value']:.4f}  "
              f"dAcc {dd['accuracy']:+.3f} {tuple(round(x, 3) for x in dd['accuracy_ci95'])}  -> {r['verdict']}")


if __name__ == "__main__":
    main()
