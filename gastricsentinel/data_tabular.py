"""TCGA-STAD clinical / genomic loading and a leakage-safe preprocessor.

All statistics (imputation values, z-score means/stds, the top-variance gene
list) are learned in `fit()` on training patients only, so they can sit inside a
sklearn Pipeline and be re-fit per cross-validation fold. The original code
selected genes and z-scored over *all* patients before splitting.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CLINICAL_FEATURES, GENE_PREFIX, MIN_GENOMIC_COVERAGE

_STAGE_RX = re.compile(r"stage\s*(iv|iii|ii|i)", re.I)
_STAGE_MAP = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
_NA = ["'--", "--", "", "Not Reported", "not reported", "NA", "NaN"]


def parse_stage(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) if 1 <= float(v) <= 4 else np.nan
    s = str(v).strip()
    if s.isdigit() and 1 <= int(s) <= 4:
        return float(s)
    m = _STAGE_RX.search(s)
    if m:
        return float(_STAGE_MAP[m.group(1).lower()])
    if s.lower() in _STAGE_MAP:
        return float(_STAGE_MAP[s.lower()])
    return np.nan


def parse_sex(v):
    s = str(v).strip().lower() if v is not None else ""
    if s in ("female", "f", "1"):
        return "female"
    if s in ("male", "m", "0"):
        return "male"
    return None


# ---------------------------------------------------------------- loaders
def load_clinical(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", na_values=_NA, low_memory=False)
    sex_col = next((c for c in ("demographic.sex_at_birth", "demographic.gender") if c in df.columns), None)
    need = {"cases.submitter_id": "patient_id", "demographic.age_at_index": "age",
            "diagnoses.ajcc_pathologic_stage": "stage"}
    missing = [c for c in need if c not in df.columns]
    if missing or sex_col is None:
        raise KeyError(f"clinical file lacks columns {missing or ['sex']}. "
                       f"Columns containing 'age'/'stage'/'sex'/'gender': "
                       f"{[c for c in df.columns if re.search('age|stage|sex|gender', c, re.I)][:20]}")
    df = df.rename(columns={**need, sex_col: "sex"})[["patient_id", "age", "sex", "stage"]]
    df["patient_id"] = df["patient_id"].astype(str).str[:12]
    df = df.groupby("patient_id", as_index=True).first()  # first non-null per column
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = df["sex"].map(parse_sex)
    df["stage"] = df["stage"].map(parse_stage)
    return df


def load_expression(path) -> pd.DataFrame:
    """UCSC Xena style matrix (genes x samples) -> patients x genes (primary tumour only)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df[~df.index.duplicated(keep="first")].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in df.columns if len(c) >= 15 and c[13:15] == "01"]
    if not keep:
        raise ValueError("no primary-tumour (-01) samples found in expression matrix")
    df = df[keep].T
    df.index = [c[:12] for c in df.index]
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [GENE_PREFIX + str(g) for g in df.columns]
    return df


def load_subtypes(path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", na_values=_NA)
    df = df[["Patient ID", "Subtype"]].dropna().drop_duplicates("Patient ID")
    return pd.Series(df["Subtype"].values, index=df["Patient ID"].astype(str).str[:12], name="subtype")


def build_patient_table(clinical_tsv, expression_path, subtypes_path=None) -> pd.DataFrame:
    """Inner-join clinical and expression by 12-char patient barcode."""
    clin = load_clinical(clinical_tsv)
    expr = load_expression(expression_path)
    df = clin.join(expr, how="inner")
    if subtypes_path:
        st = load_subtypes(subtypes_path)
        df = df.join(st, how="left")
    df.index.name = "patient_id"
    return df


# ------------------------------------------------------------ preprocessor
class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Clinical (6 dims) [+ top-variance genes] -> float32 matrix."""

    def __init__(self, include_genomic: bool = True, top_genes: int = 500):
        self.include_genomic = include_genomic
        self.top_genes = top_genes

    # -- helpers
    @staticmethod
    def _clin_cols(X):
        age = pd.to_numeric(X["age"], errors="coerce") if "age" in X else pd.Series(np.nan, index=X.index)
        stage = X["stage"].map(parse_stage) if "stage" in X else pd.Series(np.nan, index=X.index)
        sex = X["sex"].map(parse_sex) if "sex" in X else pd.Series(None, index=X.index, dtype=object)
        return age, sex, stage

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        age, _, stage = self._clin_cols(X)
        self.age_med_ = float(np.nan_to_num(age.median(), nan=0.0))
        self.age_mu_ = float(np.nan_to_num(age.fillna(self.age_med_).mean(), nan=0.0))
        self.age_sd_ = float(age.fillna(self.age_med_).std(ddof=0) or 1.0)
        self.stage_med_ = float(np.nan_to_num(stage.median(), nan=0.0))
        self.stage_mu_ = float(np.nan_to_num(stage.fillna(self.stage_med_).mean(), nan=0.0))
        self.stage_sd_ = float(stage.fillna(self.stage_med_).std(ddof=0) or 1.0)
        self.genes_, self.g_mu_, self.g_sd_ = [], np.zeros(0), np.ones(0)
        if self.include_genomic:
            gcols = [c for c in X.columns if str(c).startswith(GENE_PREFIX)]
            if not gcols:
                raise ValueError("include_genomic=True but no 'g__' gene columns present")
            G = X[gcols].apply(pd.to_numeric, errors="coerce")
            var = G.var(axis=0, skipna=True).fillna(0.0)
            keep = var.sort_values(ascending=False).head(self.top_genes).index.tolist()
            self.genes_ = [c[len(GENE_PREFIX):] for c in keep]
            self.g_mu_ = G[keep].mean(axis=0).fillna(0.0).to_numpy(dtype=np.float64)
            sd = np.array(G[keep].std(axis=0, ddof=0).fillna(1.0), dtype=np.float64)  # writable copy
            sd[sd == 0] = 1.0
            self.g_sd_ = sd
        self.feature_names_ = list(CLINICAL_FEATURES) + [f"gene:{g}" for g in self.genes_]
        return self

    def transform_parts(self, X):
        """Return (clinical[n,6], genomic[n,G]) as float32."""
        X = pd.DataFrame(X)
        age, sex, stage = self._clin_cols(X)
        am, sm, tm = age.isna(), sex.isna(), stage.isna()
        clin = np.column_stack([
            ((age.fillna(self.age_med_) - self.age_mu_) / self.age_sd_).to_numpy(float),
            am.to_numpy(float),
            (sex == "female").to_numpy(float),
            sm.to_numpy(float),
            ((stage.fillna(self.stage_med_) - self.stage_mu_) / self.stage_sd_).to_numpy(float),
            tm.to_numpy(float),
        ]).astype(np.float32)
        if not self.include_genomic:
            return clin, np.zeros((len(X), 0), dtype=np.float32)
        G = X.reindex(columns=[GENE_PREFIX + g for g in self.genes_]).apply(pd.to_numeric, errors="coerce")
        arr = G.to_numpy(dtype=np.float64)
        miss = np.isnan(arr)
        arr = np.where(miss, self.g_mu_[None, :], arr)  # missing -> train mean -> z = 0
        z = ((arr - self.g_mu_) / self.g_sd_).astype(np.float32)
        return clin, z

    def transform(self, X):
        clin, gen = self.transform_parts(X)
        return np.hstack([clin, gen]).astype(np.float32)

    # -- inference helper
    def make_row(self, age=None, sex=None, stage=None, genes: dict | None = None):
        """Build a one-row frame from user inputs; returns (frame, genomic_coverage)."""
        row = {"age": age if age not in ("", None) else np.nan, "sex": sex, "stage": stage}
        cov = 0.0
        if self.include_genomic and self.genes_:
            genes = genes or {}
            have = 0
            for g in self.genes_:
                v = genes.get(g)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = np.nan
                row[GENE_PREFIX + g] = v
                have += int(not np.isnan(v))
            cov = have / len(self.genes_)
        return pd.DataFrame([row]), cov

    @property
    def min_coverage(self):
        return MIN_GENOMIC_COVERAGE
