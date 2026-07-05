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
# paraphrased near-duplicates. Configurable per-operation up to 1.0 (100%),
# where 100% means "don't reject anything — keep even complete duplicates."
DUPLICATE_SIMILARITY_THRESHOLD = 0.98
MIN_SIMILARITY_THRESHOLD = 0.50
MAX_SIMILARITY_THRESHOLD = 1.00

# Independent of whatever storage-level threshold an admin configures (which can be
# relaxed all the way to 100% to intentionally allow duplicates into the bank), a
# single EXAM must never present the same or a near-identical question twice. This
# fixed threshold is used only when assembling an exam's question set and is not
# exposed as a user-configurable setting.
EXAM_DEDUP_THRESHOLD = 0.98


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
    cert_id: int, db_path: str = DB_PATH, exclude_id: Optional[int] = None, language: Optional[str] = None
) -> list:
    """
    Fetch [(id, normalized_text), ...] for questions banked under a certification.
    If `language` is given, only questions in that language are considered — comparing
    across languages for duplicate-detection purposes doesn't make sense (the text will
    naturally differ) and would waste comparison time.
    """
    with get_connection(db_path) as conn:
        if language:
            rows = conn.execute(
                "SELECT id, question_text FROM questions WHERE cert_id = ? AND language = ?",
                (cert_id, language),
            ).fetchall()
        else:
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

    A `threshold` of 1.0 (100%) disables duplicate detection entirely — every
    candidate is treated as unique, so even exact, complete duplicates are allowed
    through. This is the single choke point all higher-level dedup logic in this
    module routes through, so setting 100% anywhere (batch import, AI generation,
    manual add, or the cleanup tool) consistently means "keep complete duplicates."

    Uses difflib.SequenceMatcher for a robust text-similarity score (catches exact
    duplicates, minor rewording, and reordered-but-equivalent phrasing), with two
    cheap pre-filters — a length-ratio check and quick_ratio() — so the expensive
    exact ratio() computation only runs on plausible candidates.
    """
    if threshold >= 1.0:
        return None

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


def deduplicate_question_list(questions: list, threshold: float = EXAM_DEDUP_THRESHOLD) -> list:
    """
    Given a list of question dicts (e.g. an assembled exam attempt), return a new list
    with any pairwise near-duplicates removed, keeping the first occurrence of each
    group. This is the final safety net applied when assembling a single exam so it
    never shows the same or a near-identical question twice — independent of, and in
    addition to, whatever similarity threshold was used when the questions were
    originally stored (which may have been relaxed to 100% to intentionally allow
    duplicates into the bank for other purposes).
    """
    kept = []
    kept_norms = []
    for q in questions:
        match = find_near_duplicate(q["question_text"], kept_norms, threshold=threshold)
        if match:
            continue
        kept.append(q)
        kept_norms.append((q.get("id"), _normalize_text(q["question_text"])))
    return kept


def deduplicate_certification(
    cert_id: int, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD, db_path: str = DB_PATH
) -> dict:
    """
    Scan every question already banked under a certification and remove near-duplicates,
    keeping the earliest-inserted copy of each group. Comparisons are scoped within each
    language separately (an English and a Japanese question are never considered
    duplicates of each other, regardless of topic overlap). Returns:
        {"removed": n, "kept": n, "removed_ids": [...]}
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, question_text, language FROM questions WHERE cert_id = ? ORDER BY id ASC", (cert_id,)
        ).fetchall()

    kept_by_language: dict = {}
    removed_ids = []
    for row in rows:
        lang = row["language"] or "en"
        kept_for_lang = kept_by_language.setdefault(lang, [])
        match = find_near_duplicate(row["question_text"], kept_for_lang, threshold=threshold)
        if match:
            removed_ids.append(row["id"])
        else:
            kept_for_lang.append((row["id"], _normalize_text(row["question_text"])))

    total_kept = sum(len(v) for v in kept_by_language.values())

    if removed_ids:
        with get_connection(db_path) as conn:
            conn.executemany("DELETE FROM questions WHERE id = ?", [(i,) for i in removed_ids])

    return {"removed": len(removed_ids), "kept": total_kept, "removed_ids": removed_ids}


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
                language        TEXT NOT NULL DEFAULT 'en',    -- 'en' | 'ja' — UI language this question is written in
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
        # Migration for databases created before the `language` column existed:
        # add it (SQLite requires ALTER TABLE for this on older schemas) and
        # backfill every pre-existing row as 'en', since all prior content was
        # written in English. This must run BEFORE creating any index that
        # references the column, since CREATE TABLE IF NOT EXISTS above is a
        # no-op on a table that already exists without it.
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(questions)").fetchall()}
        if "language" not in existing_columns:
            conn.execute("ALTER TABLE questions ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_cert_lang ON questions(cert_id, language)")


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


def get_question_count(cert_id: int, db_path: str = DB_PATH, language: Optional[str] = None) -> int:
    """
    Number of questions banked for a certification (across all exam sets).
    If `language` is given, counts only questions in that language — the 500-question
    cap and exam-sampling pool size are tracked per (certification, language), so
    adding English questions never eats into a certification's Japanese capacity or
    vice versa.
    """
    with get_connection(db_path) as conn:
        if language:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM questions WHERE cert_id = ? AND language = ?",
                (cert_id, language),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM questions WHERE cert_id = ?", (cert_id,)
            ).fetchone()
        return row["cnt"] if row else 0


def get_question_counts_all(db_path: str = DB_PATH) -> list[dict]:
    """Question counts per certification (with an en/ja breakdown), for the admin dashboard."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.code, c.name, c.level, c.total_questions_per_exam AS cap,
                   COUNT(q.id) AS question_count,
                   SUM(CASE WHEN q.language = 'en' THEN 1 ELSE 0 END) AS en_count,
                   SUM(CASE WHEN q.language = 'ja' THEN 1 ELSE 0 END) AS ja_count
            FROM certifications c
            LEFT JOIN questions q ON q.cert_id = c.id
            GROUP BY c.id
            ORDER BY c.name
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_random_questions(cert_id: int, limit: int, db_path: str = DB_PATH, language: Optional[str] = None) -> list[dict]:
    """
    Fetch a random sample of up to `limit` questions for a given certification.

    If `language` is given ('en' or 'ja'), only questions written in that language are
    considered — this is how the exam-taking flow guarantees a learner using the
    Japanese UI is never served an English question (or vice versa).

    Guarantees that no two questions in the returned set are near-duplicates of each
    other (>= EXAM_DEDUP_THRESHOLD similar), even if such pairs exist elsewhere in the
    certification's overall bank — e.g. legacy data from before deduplication existed,
    or bulk imports/generations where the storage-level similarity threshold was
    intentionally relaxed to 100% to allow duplicates in. This guarantee always
    applies to exam sampling, independent of whatever threshold was used when the
    questions were stored, and is not user-configurable.

    If the certification's deduped pool has fewer than `limit` sufficiently distinct
    questions, returns as many as it safely can rather than padding with duplicates.
    """
    with get_connection(db_path) as conn:
        if language:
            rows = conn.execute(
                "SELECT * FROM questions WHERE cert_id = ? AND language = ? ORDER BY RANDOM()",
                (cert_id, language),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM questions WHERE cert_id = ? ORDER BY RANDOM()",
                (cert_id,),
            ).fetchall()

    selected = []
    selected_norms = []
    for row in rows:
        if len(selected) >= limit:
            break
        q = dict(row)
        if find_near_duplicate(q["question_text"], selected_norms, threshold=EXAM_DEDUP_THRESHOLD):
            continue  # too similar to a question already picked for this exam attempt
        selected.append(q)
        selected_norms.append((q["id"], _normalize_text(q["question_text"])))
    return selected


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
    language: str = "en",
    db_path: str = DB_PATH,
    check_duplicates: bool = True,
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    existing_texts: Optional[list] = None,
) -> int:
    """
    Insert a single question. Returns the new row id.

    `language` ('en' or 'ja') records which UI language this question's text is written
    in, so exam sampling can serve only questions matching the learner's selected
    language.

    Raises ValueError on invalid input, or DuplicateQuestionError (a ValueError subclass)
    if `check_duplicates` is True and the question is >= `similarity_threshold` similar to
    an already-banked question in the SAME language for the same certification.

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
    language = (language or "en").strip().lower()

    if check_duplicates:
        candidates = existing_texts if existing_texts is not None else _load_existing_normalized_texts(
            cert_id, db_path=db_path, language=language
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
                 correct_option, explanation, domain, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cert_id, exam_number, question_text, option_a, option_b, option_c, option_d,
                correct_option, explanation, domain, language,
            ),
        )
        new_id = cur.lastrowid

    if existing_texts is not None:
        existing_texts.append((new_id, _normalize_text(question_text)))

    return new_id


def batch_import_questions(
    cert_id: int, records: list[dict], db_path: str = DB_PATH,
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    default_language: str = "en",
) -> dict:
    """
    Batch-insert questions from a list of dicts (already parsed from CSV/JSON), rejecting
    any record that's a near-duplicate (>= `similarity_threshold` similar) of an
    already-banked question in the same language, or of an earlier record in this same
    import. Pass 1.0 (100%) to disable the check entirely and allow complete duplicates
    through.

    Each record may include its own `language` field ('en' or 'ja'); records without one
    use `default_language`. This lets a single import mix languages if needed, while
    keeping duplicate-detection scoped correctly within each language.

    Returns: {"inserted": n, "duplicates_skipped": n, "errors": ["row 3: reason", ...]}
    """
    inserted = 0
    duplicates_skipped = 0
    errors = []
    existing_texts_by_lang: dict = {}

    for i, rec in enumerate(records, start=1):
        try:
            missing = [f for f in REQUIRED_QUESTION_FIELDS if not str(rec.get(f, "")).strip()]
            if missing:
                raise ValueError(f"missing field(s): {', '.join(missing)}")
            language = str(rec.get("language") or default_language).strip().lower()
            if language not in existing_texts_by_lang:
                existing_texts_by_lang[language] = _load_existing_normalized_texts(
                    cert_id, db_path=db_path, language=language
                )
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
                language=language,
                db_path=db_path,
                existing_texts=existing_texts_by_lang[language],
                similarity_threshold=similarity_threshold,
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


# Human-readable names used inside the generation prompt (kept independent of the
# UI-facing i18n module so db_utils has no dependency on the Streamlit layer).
LANGUAGE_NAMES_FOR_PROMPT = {"en": "English", "ja": "Japanese (日本語)"}


def _existing_question_signatures(cert_id: int, db_path: str = DB_PATH, language: Optional[str] = None) -> set:
    """Lightweight signatures (lowercased, truncated) used only to nudge the model's prompt
    away from stems it's already produced — the actual duplicate *rejection* is handled by
    find_near_duplicate()/add_question(), which does a real similarity comparison. Scoped
    to `language` so, e.g., a Japanese generation run isn't shown English stems (which
    would just be noise for a model asked to write in Japanese)."""
    with get_connection(db_path) as conn:
        if language:
            rows = conn.execute(
                "SELECT question_text FROM questions WHERE cert_id = ? AND language = ?",
                (cert_id, language),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT question_text FROM questions WHERE cert_id = ?", (cert_id,)
            ).fetchall()
        return {r["question_text"].strip().lower()[:80] for r in rows}


_GENERATION_PROMPT_TEMPLATE = """You are an expert AWS certification exam item-writer.

Generate {batch_size} unique, realistic, scenario-based multiple-choice practice questions
for the "{cert_name}" ({cert_code}) certification exam, at the difficulty level of the real
exam. Cover a healthy variety of exam domains/topics; do not repeat the same scenario twice.

Write the question text, all four answer options, and the explanation ENTIRELY in
{language_name}. Do not mix languages within a single question — every field for every
question must be in {language_name} only.

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
    language: str = "en",
) -> dict:
    """
    Dynamically generate up to `num_questions` new questions for `cert` using the selected
    model provider (Anthropic Cloud API, a local Ollama instance, or a local Open WebUI
    instance) and persist them to the local SQLite question bank (capped at
    MAX_QUESTIONS_PER_CERT questions per certification PER LANGUAGE).

    `language` ('en' or 'ja') is passed to the model as an explicit instruction to write
    every field of every generated question in that language, and is stored on each
    inserted row so exam sampling can later filter to just that language.

    Every candidate question is checked for near-duplication (>= `similarity_threshold`
    similarity via difflib) against both the existing same-language bank and everything
    already inserted earlier in this same generation run, so the model repeating itself
    within or across batches doesn't slip duplicates into the database.

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

    language = (language or "en").strip().lower()
    language_name = LANGUAGE_NAMES_FOR_PROMPT.get(language, "English")

    current_count = get_question_count(cert["id"], db_path=db_path, language=language)
    room_available = max(0, MAX_QUESTIONS_PER_CERT - current_count)
    target = min(num_questions, room_available)

    inserted = 0
    duplicates_skipped = 0
    errors = []
    batches_run = 0
    inserted_questions = []
    # Both scoped to `language`: showing a Japanese generation run English stems (or
    # vice versa) would just be noise, and dedup across languages is meaningless since
    # the text will always differ.
    prompt_hint_stems = _existing_question_signatures(cert["id"], db_path=db_path, language=language)
    existing_texts = _load_existing_normalized_texts(cert["id"], db_path=db_path, language=language)

    remaining = target
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        existing_stems = "\n".join(f"- {s}" for s in list(prompt_hint_stems)[-40:]) or "(none yet)"
        prompt = _GENERATION_PROMPT_TEMPLATE.format(
            batch_size=this_batch,
            cert_name=cert["name"],
            cert_code=cert["code"],
            existing_stems=existing_stems,
            language_name=language_name,
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
                    language=language,
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
