import os
import uuid
from datetime import datetime, timezone

# ── Supabase credentials ─────────────────────────────────────────────────────
# SUPABASE_URL  : your project URL  (hardcoded below, or override via env var)
# SUPABASE_KEY  : service-role key  (set as environment variable — never commit it)

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://xqobdsessewpfvzoqngj.supabase.co"
)
SUPABASE_KEY = os.getenv("SUPABASE_KEY")   # set in your .env / hosting secrets

if not SUPABASE_KEY:
    raise EnvironmentError(
        "[DB] SUPABASE_KEY environment variable is not set.\n"
        "Go to Supabase → Settings → API → service_role key and add it to your .env file:\n"
        "  SUPABASE_KEY=your_service_role_key_here"
    )

# ── Install check ─────────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError(
        "[DB] supabase-py is not installed.\n"
        "Run:  pip install supabase"
    )

# ── Create client ─────────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("[DB] Connected to Supabase ✓")


# ── Utility ───────────────────────────────────────────────────────────────────
def _now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _next_patient_id() -> str:
    """
    Generate the next sequential patient id (P-0001, P-0002 …).
    Reads the highest existing id from the patients table.
    """
    res = (
        supabase.table("patients")
        .select("id")
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if res.data:
        last_num = int(res.data[0]["id"].replace("P-", ""))
        return f"P-{last_num + 1:04d}"
    return "P-0001"


# ── patients ──────────────────────────────────────────────────────────────────

def add_patient(
    name: str,
    age: int,
    gender: str,
    condition: str = "",
    risk: str = "low",
    phone: str = "",
    notes: str = "",
) -> str:
    """
    Insert a new patient row into Supabase and return its generated id.
    Matches the same signature as the old MongoDB version so the rest of
    your Flask app (app.py) requires zero changes.
    """
    new_id = _next_patient_id()
    row = {
        "id":             new_id,
        "name":           name.strip(),
        "age":            int(age),
        "gender":         gender,
        "last_diagnosis": condition or None,
        "risk":           risk,
        "risk_score":     None,
        "tier":           None,
        "last_scan":      datetime.now(timezone.utc).date().isoformat(),
    }
    res = supabase.table("patients").insert(row).execute()
    if not res.data:
        raise RuntimeError(f"[DB] add_patient failed: {res}")
    print(f"[DB] Patient inserted: {new_id} — {name}")
    return new_id


def get_all_patients() -> list:
    """Return all patients ordered by most recently updated."""
    res = (
        supabase.table("patients")
        .select("*")
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


def get_patient_by_id(patient_id: str) -> dict | None:
    """Return a single patient row or None if not found."""
    res = (
        supabase.table("patients")
        .select("*")
        .eq("id", patient_id)
        .single()
        .execute()
    )
    return res.data


def update_patient(patient_id: str, updates: dict) -> dict | None:
    """
    Update any fields on a patient row.
    Example: update_patient("P-0001", {"risk": "high", "last_diagnosis": "Adenocarcinoma"})
    """
    res = (
        supabase.table("patients")
        .update(updates)
        .eq("id", patient_id)
        .execute()
    )
    return res.data[0] if res.data else None


def delete_patient(patient_id: str) -> bool:
    """Delete a patient by id. Returns True on success."""
    res = (
        supabase.table("patients")
        .delete()
        .eq("id", patient_id)
        .execute()
    )
    return True


# ── scans ─────────────────────────────────────────────────────────────────────

def save_scan(patient_id: str, report: dict, image_path: str = None) -> str | None:
    """
    Persist an AI scan result and its 8-class probabilities.
    Returns the new scan UUID, or None on failure.
    """
    row = {
        "patient_id":      patient_id or None,
        "diagnosis":       report.get("diagnosis"),
        "recommendation":  report.get("recommendation"),
        "predicted_class": report.get("predicted_class"),
        "tier":            report.get("tier", "NEGATIVE"),
        "risk_score":      report.get("risk_score"),
        "confidence":      report.get("confidence"),
        "scan_image_path": image_path or None,
        "gradcam_path":    report.get("gradcam_url") or report.get("gradcam_path"),
        "doctor_name":     report.get("doctor_name"),
        "hospital":        report.get("hospital"),
    }
    res = supabase.table("scans").insert(row).execute()
    if not res.data:
        print(f"[DB] save_scan failed: {res}")
        return None

    scan_id = res.data[0]["id"]

    # ── Save 8-class probabilities if present ────
    probs = report.get("probabilities")
    if probs and isinstance(probs, dict):
        prob_row = {
            "scan_id": scan_id,
            "tum":  probs.get("TUM") or probs.get("tum") or 0,
            "str":  probs.get("STR") or probs.get("str") or 0,
            "lym":  probs.get("LYM") or probs.get("lym") or 0,
            "deb":  probs.get("DEB") or probs.get("deb") or 0,
            "muc":  probs.get("MUC") or probs.get("muc") or 0,
            "mus":  probs.get("MUS") or probs.get("mus") or 0,
            "norm": probs.get("NORM") or probs.get("norm") or 0,
            "adi":  probs.get("ADI") or probs.get("adi") or 0,
        }
        supabase.table("scan_probabilities").insert(prob_row).execute()

    # ── Update the patient's risk level to match latest scan ────
    if patient_id and report.get("risk_score") is not None:
        score = report["risk_score"]
        risk  = "high" if score >= 70 else "mid" if score >= 40 else "low"
        update_patient(patient_id, {
            "risk":           risk,
            "risk_score":     score,
            "tier":           report.get("tier"),
            "last_diagnosis": report.get("diagnosis"),
            "last_scan":      datetime.now(timezone.utc).date().isoformat(),
        })

    print(f"[DB] Scan saved: {scan_id}")
    return scan_id


def get_patient_scans(patient_id: str) -> list:
    """Return all scan rows for a patient, newest first."""
    res = (
        supabase.table("scans")
        .select("*, scan_probabilities(*)")
        .eq("patient_id", patient_id)
        .order("scanned_at", desc=True)
        .execute()
    )
    return res.data or []


def get_all_scans() -> list:
    """Return every scan row, newest first."""
    res = (
        supabase.table("scans")
        .select("*")
        .order("scanned_at", desc=True)
        .execute()
    )
    return res.data or []


# ── feedback ──────────────────────────────────────────────────────────────────

def save_feedback(
    prediction: str,
    correction: str,
    scan_id: str = None,
    patient_id: str = None,
    rating: int = None,
    notes: str = None,
    doctor_name: str = None,
) -> None:
    """
    Store doctor feedback on a prediction.
    `prediction` maps to verdict='confirm', anything else to verdict='incorrect'.
    """
    verdict = "confirm" if str(prediction).lower() in ("confirm", "correct", "true") else "incorrect"
    row = {
        "scan_id":     scan_id    or None,
        "patient_id":  patient_id or None,
        "doctor_name": doctor_name or None,
        "verdict":     verdict,
        "rating":      int(rating) if rating is not None else 3,
        "notes":       notes or correction or None,
    }
    res = supabase.table("feedback").insert(row).execute()
    if not res.data:
        print(f"[DB] save_feedback failed: {res}")


def get_all_feedback() -> list:
    """Return all feedback rows, newest first."""
    res = (
        supabase.table("feedback")
        .select("*")
        .order("submitted_at", desc=True)
        .execute()
    )
    return res.data or []


# ── clinical / genomic data ───────────────────────────────────────────────────

def save_clinical_data(
    patient_id: str,
    scan_id: str = None,
    age: int = None,
    gender: str = None,
    stage: str = None,
    gene_score: float = None,
    genomic_risk: float = None,
) -> None:
    """Save clinical and genomic inputs used in the multimodal fusion model."""
    row = {
        "patient_id":   patient_id,
        "scan_id":      scan_id      or None,
        "age":          int(age)     if age          is not None else None,
        "gender":       gender       or None,
        "stage":        stage        or None,
        "gene_score":   float(gene_score)    if gene_score    is not None else None,
        "genomic_risk": float(genomic_risk)  if genomic_risk  is not None else None,
    }
    supabase.table("clinical_data").insert(row).execute()


# ── stats helper (used by /stats Flask route) ─────────────────────────────────

def get_stats() -> dict:
    """
    Return aggregate counts used by the dashboard.
    Equivalent to the old MongoDB aggregation queries.
    """
    patients = get_all_patients()
    total    = len(patients)
    high     = sum(1 for p in patients if (p.get("risk") or "").lower() == "high")
    mid      = sum(1 for p in patients if (p.get("risk") or "").lower() in ("mid", "medium"))
    low      = total - high - mid

    scans_res = supabase.table("scans").select("id", count="exact").execute()
    total_scans = scans_res.count or 0

    return {
        "total_patients": total,
        "total_scans":    total_scans,
        "high_risk":      high,
        "mid_risk":       mid,
        "low_risk":       low,
    }


# ── legacy compatibility shim ─────────────────────────────────────────────────
# Any Flask route that called get_db().patients.find() etc. can keep working
# by calling get_db() — the returned object wraps the Supabase helpers.

class _SupabaseDB:
    """
    Thin compatibility wrapper so existing Flask routes that use
    `db = get_db(); db.patients.find(...)` keep working without changes.
    """
    class _Table:
        def __init__(self, name: str):
            self._name = name

        def find(self, query: dict = None, projection: dict = None) -> list:
            res = supabase.table(self._name).select("*").execute()
            rows = res.data or []
            if query:
                rows = [r for r in rows if all(r.get(k) == v for k, v in query.items() if not k.startswith("$"))]
            return rows

        def find_one(self, query: dict = None) -> dict | None:
            rows = self.find(query)
            return rows[0] if rows else None

        def insert_one(self, doc: dict):
            res = supabase.table(self._name).insert(doc).execute()
            return type("R", (), {"inserted_id": res.data[0].get("id") if res.data else None})()

        def update_one(self, query: dict, update: dict):
            if "$set" not in update:
                return
            filters = {k: v for k, v in query.items() if not k.startswith("$")}
            q = supabase.table(self._name).update(update["$set"])
            for k, v in filters.items():
                q = q.eq(k, v)
            q.execute()

        def delete_one(self, query: dict):
            filters = {k: v for k, v in query.items() if not k.startswith("$")}
            q = supabase.table(self._name).delete()
            for k, v in filters.items():
                q = q.eq(k, v)
            q.execute()

        def count_documents(self, query: dict = None) -> int:
            res = supabase.table(self._name).select("id", count="exact").execute()
            return res.count or 0

    def __init__(self):
        self.patients = _SupabaseDB._Table("patients")
        self.scans    = _SupabaseDB._Table("scans")
        self.feedback = _SupabaseDB._Table("feedback")


_db_instance = _SupabaseDB()


def get_db() -> _SupabaseDB:
    """Return a db handle compatible with the old MongoDB usage pattern."""
    return _db_instance