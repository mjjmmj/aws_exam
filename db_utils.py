"""
db_utils.py
------------
Centralized data-access layer for the AWS Mock Exam Engine.

All SQLite interactions (reads, writes, batch imports) are funneled through
this module so that app.py and init_db.py never touch raw SQL directly.
This keeps a clean separation of concerns between the UI layer and the
persistence layer.
"""

import sqlite3
import json
import csv
import io
import os
import re
import difflib
from contextlib import contextmanager
from typing import Optional

import ai_providers

DB_PATH = "aws_exams.db"

# A candidate question is rejected if it's at or above this similarity ratio
# (difflib SequenceMatcher, 0.0-1.0) to any already-banked question for the
# same certification. This blocks exact duplicates as well as closely
# paraphrased near-duplicates.
DUPLICATE_SIMILARITY_THRESHOLD = 0.90


class DuplicateQuestionError(ValueError):
    """Raised when a candidate question is a near-duplicate of an existing banked question."""

    def __init__(self, message: str, existing_id: Optional[int] = None, similarity: Optional[float] = None):
        super().__init__(message)
        self.existing_id = existing_id
        self.similarity = similarity


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting differences don't affect comparison."""
    return " ".join((text or "").strip().lower().split())


def _load_existing_normalized_texts(
    cert_id: int, db_path: str = DB_PATH, exclude_id: Optional[int] = None
) -> list:
    """Fetch [(id, normalized_text), ...] for every question banked under a certification."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, question_text FROM questions WHERE cert_id = ?", (cert_id,)
        ).fetchall()
    return [(r["id"], _normalize_text(r["question_text"])) for r in rows if r["id"] != exclude_id]


def find_near_duplicate(
    question_text: str,
    existing_texts: list,
    threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
):
    """
    Compare `question_text` against a list of (id, normalized_text) candidates.
    Returns (matching_id, similarity_ratio) for the first match at/above `threshold`,
    or None if no near-duplicate is found.

    Uses difflib.SequenceMatcher for a robust text-similarity score (catches exact
    duplicates, minor rewording, and reordered-but-equivalent phrasing), with two
    cheap pre-filters — a length-ratio check and quick_ratio() — so the expensive
    exact ratio() computation only runs on plausible candidates.
    """
    norm_target = _normalize_text(question_text)
    if not norm_target:
        return None
    target_len = len(norm_target)

    for existing_id, norm_existing in existing_texts:
        if not norm_existing:
            continue
        existing_len = len(norm_existing)
        longer, shorter = max(target_len, existing_len), min(target_len, existing_len)
        if longer == 0:
            continue
        # Two strings whose lengths differ too much cannot reach a high similarity
        # ratio; skip the expensive comparison. Small buffer since length ratio is
        # only a loose proxy for SequenceMatcher's actual ratio.
        if shorter / longer < threshold - 0.10:
            continue
        matcher = difflib.SequenceMatcher(None, norm_target, norm_existing, autojunk=False)
        if matcher.quick_ratio() < threshold:
            continue
        ratio = matcher.ratio()
        if ratio >= threshold:
            return existing_id, ratio
    return None


def deduplicate_certification(
    cert_id: int, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD, db_path: str = DB_PATH
) -> dict:
    """
    Scan every question already banked under a certification and remove near-duplicates,
    keeping the earliest-inserted copy of each group. Returns:
        {"removed": n, "kept": n, "removed_ids": [...]}
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, question_text FROM questions WHERE cert_id = ? ORDER BY id ASC", (cert_id,)
        ).fetchall()

    kept: list = []
    removed_ids = []
    for row in rows:
        match = find_near_duplicate(row["question_text"], kept, threshold=threshold)
        if match:
            removed_ids.append(row["id"])
        else:
            kept.append((row["id"], _normalize_text(row["question_text"])))

    if removed_ids:
        with get_connection(db_path) as conn:
            conn.executemany("DELETE FROM questions WHERE id = ?", [(i,) for i in removed_ids])

    return {"removed": len(removed_ids), "kept": len(kept), "removed_ids": removed_ids}


def deduplicate_all_certifications(
    threshold: float = DUPLICATE_SIMILARITY_THRESHOLD, db_path: str = DB_PATH
) -> dict:
    """Run deduplicate_certification for every certification. Returns {cert_code: result_dict}."""
    results = {}
    for cert in get_certifications(db_path=db_path):
        results[cert["code"]] = deduplicate_certification(cert["id"], threshold=threshold, db_path=db_path)
    return results


VALID_OPTIONS = {"A", "B", "C", "D"}
REQUIRED_QUESTION_FIELDS = [
    "question_text",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_option",
    "explanation",
]
VALID_LEVELS = ["Foundational", "Associate", "Professional", "Specialty"]

# Each certification's question bank is designed to scale up to this many
# questions (e.g. via the AI generator or batch import), supporting up to
# 10 full-length practice exams of ~50 questions each.
MAX_QUESTIONS_PER_CERT = 500


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Context-managed SQLite connection with foreign keys enabled and Row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(db_path: str = DB_PATH) -> None:
    """Create the certifications and questions tables if they do not already exist."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS certifications (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                code            TEXT UNIQUE NOT NULL,
                name            TEXT NOT NULL,
                level           TEXT NOT NULL,          -- 'Foundational' | 'Associate' | 'Professional' | 'Specialty'
                pass_threshold  REAL NOT NULL,           -- e.g. 0.70 or 0.75
                total_questions_per_exam INTEGER NOT NULL DEFAULT 500  -- max bank size supported for this cert
            );

            CREATE TABLE IF NOT EXISTS questions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cert_id         INTEGER NOT NULL,
                exam_number     INTEGER NOT NULL DEFAULT 1,   -- supports up to 10 full exams per cert
                question_text   TEXT NOT NULL,
                option_a        TEXT NOT NULL,
                option_b        TEXT NOT NULL,
                option_c        TEXT NOT NULL,
                option_d        TEXT NOT NULL,
                correct_option  TEXT NOT NULL CHECK (correct_option IN ('A','B','C','D')),
                explanation     TEXT NOT NULL,
                domain          TEXT,                          -- optional exam-domain tag
                FOREIGN KEY (cert_id) REFERENCES certifications(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_questions_cert ON questions(cert_id);
            CREATE INDEX IF NOT EXISTS idx_questions_cert_exam ON questions(cert_id, exam_number);
            """
        )
        # Migration for databases created before the 500-question cap was introduced:
        # raise any certification's cap that is still below the new maximum.
        conn.execute(
            "UPDATE certifications SET total_questions_per_exam = ? "
            "WHERE total_questions_per_exam < ?",
            (MAX_QUESTIONS_PER_CERT, MAX_QUESTIONS_PER_CERT),
        )


def seed_certifications(db_path: str = DB_PATH) -> None:
    """Insert the five target certifications if they don't already exist."""
    certs = [
        ("SAP-C02", "AWS Certified Solutions Architect – Professional", "Professional", 0.75, MAX_QUESTIONS_PER_CERT),
        ("DOP-C02", "AWS Certified DevOps Engineer – Professional", "Professional", 0.75, MAX_QUESTIONS_PER_CERT),
        ("MLS-C01", "AWS Certified Machine Learning – Specialty", "Specialty", 0.75, MAX_QUESTIONS_PER_CERT),
        ("AIF-C01", "AWS Certified AI Practitioner", "Foundational", 0.70, MAX_QUESTIONS_PER_CERT),
        ("CLF-C02", "AWS Certified Cloud Practitioner", "Foundational", 0.70, MAX_QUESTIONS_PER_CERT),
    ]
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO certifications (code, name, level, pass_threshold, total_questions_per_exam)
            VALUES (?, ?, ?, ?, ?)
            """,
            certs,
        )


def add_certification(
    code: str,
    name: str,
    level: str,
    pass_threshold: float,
    total_questions_per_exam: int = MAX_QUESTIONS_PER_CERT,
    db_path: str = DB_PATH,
) -> int:
    """
    Add a new, user-defined certification so it appears in the exam dropdown.
    Returns the new certification's id. Raises ValueError on invalid/duplicate input.
    """
    code = code.strip().upper()
    name = name.strip()
    if not code or not name:
        raise ValueError("Certification code and name are required.")
    if not re.match(r"^[A-Z0-9\-]{2,20}$", code):
        raise ValueError("Code must be 2-20 characters: letters, numbers, and hyphens only.")
    if level not in VALID_LEVELS:
        raise ValueError(f"Level must be one of: {', '.join(VALID_LEVELS)}")
    if not (0.0 < pass_threshold <= 1.0):
        raise ValueError("Pass threshold must be a fraction between 0 and 1 (e.g. 0.72 for 72%).")

    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT 1 FROM certifications WHERE code = ?", (code,)
        ).fetchone()
        if existing:
            raise ValueError(f"A certification with code '{code}' already exists.")
        cur = conn.execute(
            """
            INSERT INTO certifications (code, name, level, pass_threshold, total_questions_per_exam)
            VALUES (?, ?, ?, ?, ?)
            """,
            (code, name, level, pass_threshold, min(total_questions_per_exam, MAX_QUESTIONS_PER_CERT)),
        )
        return cur.lastrowid


def get_certifications(db_path: str = DB_PATH) -> list[sqlite3.Row]:
    """Return all certifications ordered by name."""
    with get_connection(db_path) as conn:
        return conn.execute("SELECT * FROM certifications ORDER BY name").fetchall()


def get_certification_by_code(code: str, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    with get_connection(db_path) as conn:
        return conn.execute(
            "SELECT * FROM certifications WHERE code = ?", (code,)
        ).fetchone()


def get_question_count(cert_id: int, db_path: str = DB_PATH) -> int:
    """Total number of questions banked for a certification (across all exam sets)."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM questions WHERE cert_id = ?", (cert_id,)
        ).fetchone()
        return row["cnt"] if row else 0


def get_question_counts_all(db_path: str = DB_PATH) -> list[dict]:
    """Question counts per certification, for the admin dashboard summary table."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.code, c.name, c.level, c.total_questions_per_exam AS cap,
                   COUNT(q.id) AS question_count
            FROM certifications c
            LEFT JOIN questions q ON q.cert_id = c.id
            GROUP BY c.id
            ORDER BY c.name
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_random_questions(cert_id: int, limit: int, db_path: str = DB_PATH) -> list[dict]:
    """Fetch a random sample of `limit` questions for a given certification."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM questions
            WHERE cert_id = ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (cert_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def add_question(
    cert_id: int,
    question_text: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
    correct_option: str,
    explanation: str,
    exam_number: int = 1,
    domain: Optional[str] = None,
    db_path: str = DB_PATH,
    check_duplicates: bool = True,
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    existing_texts: Optional[list] = None,
) -> int:
    """
    Insert a single question. Returns the new row id.

    Raises ValueError on invalid input, or DuplicateQuestionError (a ValueError subclass)
    if `check_duplicates` is True and the question is >= `similarity_threshold` similar to
    an already-banked question for the same certification.

    `existing_texts`, if provided, is a mutable [(id, normalized_text), ...] cache used
    instead of re-querying the database — callers doing many inserts in a row (batch
    import, AI generation) should pass one shared list so it can grow as each question is
    added, catching duplicates within the same batch as well as against prior data.
    """
    correct_option = correct_option.strip().upper()
    if correct_option not in VALID_OPTIONS:
        raise ValueError(f"correct_option must be one of A/B/C/D, got '{correct_option}'")
    if not all([question_text, option_a, option_b, option_c, option_d, explanation]):
        raise ValueError("All question fields (text, 4 options, explanation) are required.")

    if check_duplicates:
        candidates = existing_texts if existing_texts is not None else _load_existing_normalized_texts(
            cert_id, db_path=db_path
        )
        match = find_near_duplicate(question_text, candidates, threshold=similarity_threshold)
        if match:
            existing_id, ratio = match
            raise DuplicateQuestionError(
                f"rejected as a near-duplicate ({ratio*100:.0f}% similar to existing question id={existing_id}); "
                f"questions must be under {int(similarity_threshold*100)}% similar to already-banked ones",
                existing_id=existing_id, similarity=ratio,
            )

    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO questions
                (cert_id, exam_number, question_text, option_a, option_b, option_c, option_d,
                 correct_option, explanation, domain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cert_id, exam_number, question_text, option_a, option_b, option_c, option_d,
                correct_option, explanation, domain,
            ),
        )
        new_id = cur.lastrowid

    if existing_texts is not None:
        existing_texts.append((new_id, _normalize_text(question_text)))

    return new_id


def batch_import_questions(cert_id: int, records: list[dict], db_path: str = DB_PATH) -> dict:
    """
    Batch-insert questions from a list of dicts (already parsed from CSV/JSON), rejecting
    any record that's a near-duplicate (>= DUPLICATE_SIMILARITY_THRESHOLD similar) of an
    already-banked question or of an earlier record in this same import.

    Returns: {"inserted": n, "duplicates_skipped": n, "errors": ["row 3: reason", ...]}
    """
    inserted = 0
    duplicates_skipped = 0
    errors = []
    existing_texts = _load_existing_normalized_texts(cert_id, db_path=db_path)

    for i, rec in enumerate(records, start=1):
        try:
            missing = [f for f in REQUIRED_QUESTION_FIELDS if not str(rec.get(f, "")).strip()]
            if missing:
                raise ValueError(f"missing field(s): {', '.join(missing)}")
            add_question(
                cert_id=cert_id,
                question_text=str(rec["question_text"]).strip(),
                option_a=str(rec["option_a"]).strip(),
                option_b=str(rec["option_b"]).strip(),
                option_c=str(rec["option_c"]).strip(),
                option_d=str(rec["option_d"]).strip(),
                correct_option=str(rec["correct_option"]).strip(),
                explanation=str(rec["explanation"]).strip(),
                exam_number=int(rec.get("exam_number", 1) or 1),
                domain=rec.get("domain"),
                db_path=db_path,
                existing_texts=existing_texts,
            )
            inserted += 1
        except DuplicateQuestionError as exc:
            duplicates_skipped += 1
            errors.append(f"row {i}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the admin, not swallowed silently
            errors.append(f"row {i}: {exc}")
    return {"inserted": inserted, "duplicates_skipped": duplicates_skipped, "errors": errors}


def parse_uploaded_file(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Parse an uploaded CSV or JSON file into a list of question dicts.
    Expected columns/keys: question_text, option_a, option_b, option_c, option_d,
                            correct_option, explanation, exam_number (optional), domain (optional)
    """
    text = file_bytes.decode("utf-8-sig")
    if filename.lower().endswith(".json"):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("questions", [])
        return data
    elif filename.lower().endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .json file.")


def get_exam_numbers(cert_id: int, db_path: str = DB_PATH) -> list[int]:
    """Distinct exam_number values banked for a certification (useful for future exam-set selection)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT exam_number FROM questions WHERE cert_id = ? ORDER BY exam_number",
            (cert_id,),
        ).fetchall()
        return [r["exam_number"] for r in rows]


# --------------------------------------------------------------------------
# AI-powered dynamic question generation (provider-agnostic: Anthropic /
# local Ollama / local Open WebUI — see ai_providers.py)
# --------------------------------------------------------------------------
def is_ai_generation_available(provider: str = ai_providers.PROVIDER_ANTHROPIC,
                                base_url: str = "", api_key: str = "") -> bool:
    """True if the minimum configuration for the given provider is present."""
    return ai_providers.is_provider_configured(provider, base_url=base_url, api_key=api_key)


def _existing_question_signatures(cert_id: int, db_path: str = DB_PATH) -> set:
    """Lightweight signatures (lowercased, truncated) used only to nudge the model's prompt
    away from stems it's already produced — the actual duplicate *rejection* is handled by
    find_near_duplicate()/add_question(), which does a real similarity comparison."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT question_text FROM questions WHERE cert_id = ?", (cert_id,)
        ).fetchall()
        return {r["question_text"].strip().lower()[:80] for r in rows}


_GENERATION_PROMPT_TEMPLATE = """You are an expert AWS certification exam item-writer.

Generate {batch_size} unique, realistic, scenario-based multiple-choice practice questions
for the "{cert_name}" ({cert_code}) certification exam, at the difficulty level of the real
exam. Cover a healthy variety of exam domains/topics; do not repeat the same scenario twice.

Avoid writing questions that closely resemble any of these already-banked question stems
(do not reuse these scenarios or near-duplicates of them):
{existing_stems}

Respond with ONLY a raw JSON array (no markdown fences, no commentary) of exactly
{batch_size} objects, each with this exact shape:
{{
  "question_text": "...",
  "option_a": "...",
  "option_b": "...",
  "option_c": "...",
  "option_d": "...",
  "correct_option": "A" | "B" | "C" | "D",
  "explanation": "A thorough rationale explaining why the correct answer is right AND why each
                   distractor is wrong.",
  "domain": "short exam domain/topic tag"
}}
"""


def _parse_ai_json_array(raw_text: str) -> list[dict]:
    """Extract a JSON array from the model's response, tolerating stray markdown fences."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain a JSON array.")
    return json.loads(cleaned[start : end + 1])


def generate_questions_with_ai(
    cert: dict,
    num_questions: int,
    provider: str = ai_providers.PROVIDER_ANTHROPIC,
    model: str = "claude-sonnet-5",
    base_url: str = "",
    api_key: str = "",
    custom_path: str = "",
    exam_number: int = 1,
    batch_size: int = 10,
    progress_callback=None,
    db_path: str = DB_PATH,
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Dynamically generate up to `num_questions` new questions for `cert` using the selected
    model provider (Anthropic Cloud API, a local Ollama instance, or a local Open WebUI
    instance) and persist them to the local SQLite question bank (capped at
    MAX_QUESTIONS_PER_CERT total per certification).

    Every candidate question is checked for near-duplication (>= `similarity_threshold`
    similarity via difflib) against both the existing bank and everything already inserted
    earlier in this same generation run, so the model repeating itself within or across
    batches doesn't slip duplicates into the database.

    `progress_callback`, if provided, is called after each batch with (completed, total).

    Returns a summary dict:
        {"inserted": n, "duplicates_skipped": n, "errors": [...], "batches_run": n,
         "inserted_questions": [question_dict, ...]}
    The `inserted_questions` list lets callers (e.g. the exam-taking flow) immediately use
    the freshly generated set without a separate database query.
    """
    if not ai_providers.is_provider_configured(provider, base_url=base_url, api_key=api_key):
        raise RuntimeError(
            f"'{provider}' is not fully configured. Check the model provider settings "
            "(API key / base URL) and try again."
        )

    current_count = get_question_count(cert["id"], db_path=db_path)
    room_available = max(0, MAX_QUESTIONS_PER_CERT - current_count)
    target = min(num_questions, room_available)

    inserted = 0
    duplicates_skipped = 0
    errors = []
    batches_run = 0
    inserted_questions = []
    prompt_hint_stems = _existing_question_signatures(cert["id"], db_path=db_path)  # for prompt diversity nudging only
    existing_texts = _load_existing_normalized_texts(cert["id"], db_path=db_path)   # authoritative dedup cache

    remaining = target
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        existing_stems = "\n".join(f"- {s}" for s in list(prompt_hint_stems)[-40:]) or "(none yet)"
        prompt = _GENERATION_PROMPT_TEMPLATE.format(
            batch_size=this_batch,
            cert_name=cert["name"],
            cert_code=cert["code"],
            existing_stems=existing_stems,
        )

        try:
            raw_text = ai_providers.call_model(
                provider=provider, model=model, prompt=prompt, base_url=base_url, api_key=api_key,
                custom_path=custom_path,
            )
            records = _parse_ai_json_array(raw_text)
        except Exception as exc:  # noqa: BLE001 - surfaced to the admin
            errors.append(f"batch starting at {inserted + duplicates_skipped + 1}: generation failed — {exc}")
            batches_run += 1
            remaining -= this_batch
            if progress_callback:
                progress_callback(target - remaining, target)
            continue

        for rec in records:
            if inserted >= target:
                break  # Hard stop: never insert more than the target, even if the
                       # model returned extra records in this batch.
            try:
                missing = [f for f in REQUIRED_QUESTION_FIELDS if not str(rec.get(f, "")).strip()]
                if missing:
                    raise ValueError(f"missing field(s): {', '.join(missing)}")
                question_text = str(rec["question_text"]).strip()
                new_id = add_question(
                    cert_id=cert["id"],
                    question_text=question_text,
                    option_a=str(rec["option_a"]).strip(),
                    option_b=str(rec["option_b"]).strip(),
                    option_c=str(rec["option_c"]).strip(),
                    option_d=str(rec["option_d"]).strip(),
                    correct_option=str(rec["correct_option"]).strip(),
                    explanation=str(rec["explanation"]).strip(),
                    exam_number=exam_number,
                    domain=rec.get("domain"),
                    db_path=db_path,
                    existing_texts=existing_texts,
                    similarity_threshold=similarity_threshold,
                )
                prompt_hint_stems.add(question_text.strip().lower()[:80])
                inserted += 1
                inserted_questions.append(get_question_by_id(new_id, db_path=db_path))
            except DuplicateQuestionError:
                duplicates_skipped += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"generated record skipped: {exc}")

        batches_run += 1
        remaining -= this_batch
        if progress_callback:
            progress_callback(target - remaining, target)
        if inserted >= target:
            break
        if batches_run >= (target // max(1, batch_size) + 20):
            # Safety valve: stop if we're running far more batches than expected
            # (e.g. the model keeps returning near-duplicate content).
            errors.append("Stopped early: too many batches were needed to reach the target "
                           "(the model may be producing repetitive content).")
            break

    if num_questions > room_available:
        errors.append(
            f"Requested {num_questions}, but only {room_available} slot(s) remained "
            f"under the {MAX_QUESTIONS_PER_CERT}-question cap for this certification."
        )

    return {
        "inserted": inserted,
        "duplicates_skipped": duplicates_skipped,
        "errors": errors,
        "batches_run": batches_run,
        "inserted_questions": inserted_questions,
    }


def get_question_by_id(question_id: int, db_path: str = DB_PATH) -> Optional[dict]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
        return dict(row) if row else None
