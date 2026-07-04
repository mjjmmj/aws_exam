"""
app.py
------
Streamlit front end for the AWS Certification Mock Exam Engine.

Two views:
  1. "Take an Exam"   - the learner-facing exam experience.
  2. "Admin Dashboard" - question-bank management (manual add + batch import).

Run with:
    streamlit run app.py
"""

import os
import time
import random

import streamlit as st

import db_utils
import ai_providers
from init_db import main as run_init_db

# --------------------------------------------------------------------------
# Page config & one-time DB bootstrap
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AWS Mock Exam Engine",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.path.exists(db_utils.DB_PATH):
    run_init_db()


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "exam_stage": "not_started",   # 'not_started' | 'in_progress' | 'submitted'
        "exam_questions": [],          # list[dict] snapshot of the questions for this attempt
        "user_answers": {},            # {question_id: 'A'/'B'/'C'/'D'/None}
        "exam_cert": None,             # sqlite3.Row of the certification taken
        "exam_start_time": None,
        "timed_mode": False,
        "time_limit_minutes": 90,
        "score_summary": None,         # dict populated on submission
        # Shared AI model-provider configuration, reused by both the Admin
        # Dashboard's generator and the "Generate New with AI" exam option.
        "ai_provider": ai_providers.PROVIDER_ANTHROPIC,
        "ai_model": "",
        "ai_base_url": "",
        "ai_api_key": "",
        "ai_custom_path": "",
        "ai_available_models": [],     # cache of the last successful model listing
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --------------------------------------------------------------------------
# Shared AI provider/model configuration widget
# --------------------------------------------------------------------------
def render_ai_provider_selector(key_prefix: str):
    """
    Renders provider/base-URL/API-key/model widgets. Selections are persisted in
    st.session_state (shared across the Admin Dashboard and the exam-taking flow)
    so the user only has to configure their provider once per session.

    Returns (provider, model, base_url, api_key, custom_path) — model may be "" if
    none selected yet. custom_path is only meaningful for Open WebUI.
    """
    provider = st.selectbox(
        "Model Provider",
        ai_providers.PROVIDERS,
        index=ai_providers.PROVIDERS.index(st.session_state.ai_provider),
        key=f"{key_prefix}_provider",
        help="Use Anthropic's cloud API, or point at a local Ollama or Open WebUI instance.",
    )
    st.session_state.ai_provider = provider

    base_url = st.session_state.ai_base_url
    api_key = st.session_state.ai_api_key
    custom_path = st.session_state.ai_custom_path

    if ai_providers.provider_requires_base_url(provider):
        default_url = st.session_state.ai_base_url or ai_providers.default_base_url(provider)
        base_url = st.text_input(
            "Base URL", value=default_url, key=f"{key_prefix}_base_url",
            help="Address of your local Ollama / Open WebUI server.",
        )
        st.session_state.ai_base_url = base_url

    if ai_providers.provider_requires_api_key(provider):
        placeholder = "sk-ant-..." if provider == ai_providers.PROVIDER_ANTHROPIC else "Open WebUI API key"
        env_hint = ""
        if provider == ai_providers.PROVIDER_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
            env_hint = " (ANTHROPIC_API_KEY is set in the environment — leave blank to use it)"
        api_key = st.text_input(
            f"API Key{env_hint}", value=st.session_state.ai_api_key, type="password",
            key=f"{key_prefix}_api_key", placeholder=placeholder,
        )
        st.session_state.ai_api_key = api_key

    if provider == ai_providers.PROVIDER_OPENWEBUI:
        with st.expander("⚙️ Advanced: Custom API Path"):
            st.caption(
                "By default we auto-try common Open WebUI chat-completions paths. If your "
                "instance uses a nonstandard layout, or auto-detection fails, set the exact "
                "path here (e.g. `/api/chat/completions`)."
            )
            custom_path = st.text_input(
                "Custom chat-completions path (optional)",
                value=st.session_state.ai_custom_path,
                key=f"{key_prefix}_custom_path",
                placeholder="/api/chat/completions",
            )
            st.session_state.ai_custom_path = custom_path

    col_model, col_refresh = st.columns([3, 1])
    with col_refresh:
        st.write("")  # vertical alignment spacer
        fetch_clicked = st.button("🔄 Fetch Models", key=f"{key_prefix}_fetch_models")

    if fetch_clicked:
        try:
            with st.spinner("Fetching available models..."):
                models = ai_providers.list_models(provider, base_url=base_url, api_key=api_key)
            st.session_state.ai_available_models = models
            if not models:
                st.warning("No models were returned by this provider.")
        except RuntimeError as e:
            st.session_state.ai_available_models = []
            st.error(str(e))

    with col_model:
        available = st.session_state.ai_available_models
        if available:
            default_idx = available.index(st.session_state.ai_model) if st.session_state.ai_model in available else 0
            model = st.selectbox("Model", available, index=default_idx, key=f"{key_prefix}_model_select")
        else:
            model = st.text_input(
                "Model name",
                value=st.session_state.ai_model,
                key=f"{key_prefix}_model_text",
                help="Click 'Fetch Models' to list available models, or type one directly "
                     "(e.g. 'llama3:8b' for Ollama, 'claude-sonnet-5' for Anthropic).",
            )
        st.session_state.ai_model = model

    return provider, model, base_url, api_key, custom_path


# --------------------------------------------------------------------------
# Exam lifecycle helpers
# --------------------------------------------------------------------------
def start_exam(cert_row, num_questions: int, timed_mode: bool, time_limit_minutes: int,
                explicit_questions: list = None):
    """
    Begin a new exam attempt.

    If `explicit_questions` is provided (e.g. a freshly AI-generated set), those exact
    questions are used for this attempt. Otherwise, `num_questions` are sampled randomly
    from the existing question bank for `cert_row`.
    """
    questions = explicit_questions if explicit_questions is not None else db_utils.get_random_questions(
        cert_row["id"], num_questions
    )
    questions = list(questions)
    random.shuffle(questions)
    st.session_state.exam_questions = questions
    st.session_state.user_answers = {q["id"]: None for q in questions}
    st.session_state.exam_cert = dict(cert_row)
    st.session_state.exam_stage = "in_progress"
    st.session_state.exam_start_time = time.time()
    st.session_state.timed_mode = timed_mode
    st.session_state.time_limit_minutes = time_limit_minutes
    st.session_state.score_summary = None


def grade_exam():
    questions = st.session_state.exam_questions
    answers = st.session_state.user_answers
    total = len(questions)
    correct = sum(
        1 for q in questions if answers.get(q["id"]) == q["correct_option"]
    )
    unanswered = sum(1 for q in questions if answers.get(q["id"]) is None)
    score_pct = (correct / total) if total else 0.0
    threshold = st.session_state.exam_cert["pass_threshold"]
    passed = score_pct >= threshold

    elapsed_seconds = 0
    if st.session_state.exam_start_time:
        elapsed_seconds = int(time.time() - st.session_state.exam_start_time)

    st.session_state.score_summary = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct - unanswered,
        "unanswered": unanswered,
        "score_pct": score_pct,
        "threshold": threshold,
        "passed": passed,
        "elapsed_seconds": elapsed_seconds,
    }
    st.session_state.exam_stage = "submitted"


def reset_exam():
    st.session_state.exam_stage = "not_started"
    st.session_state.exam_questions = []
    st.session_state.user_answers = {}
    st.session_state.exam_cert = None
    st.session_state.exam_start_time = None
    st.session_state.score_summary = None


# --------------------------------------------------------------------------
# UI: Sidebar navigation + exam configuration
# --------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("☁️ AWS Mock Exam Engine")
    view = st.sidebar.radio(
        "Navigation",
        ["Take an Exam", "Admin Dashboard"],
        index=0,
        help="Switch between practicing exams and managing the question bank.",
    )
    st.sidebar.divider()

    if view == "Take an Exam":
        render_exam_config_sidebar()

    return view


def render_exam_config_sidebar():
    certifications = db_utils.get_certifications()
    if not certifications:
        st.sidebar.warning("No certifications found in the database.")
        return

    st.sidebar.subheader("Exam Configuration")

    cert_labels = [f"{c['code']} — {c['name']}" for c in certifications]
    cert_lookup = {label: cert for label, cert in zip(cert_labels, certifications)}

    default_index = 0
    if st.session_state.exam_cert:
        current_code = st.session_state.exam_cert["code"]
        for i, c in enumerate(certifications):
            if c["code"] == current_code:
                default_index = i
                break

    selected_label = st.sidebar.selectbox(
        "Target Certification", cert_labels, index=default_index,
        disabled=st.session_state.exam_stage == "in_progress",
    )
    selected_cert = dict(cert_lookup[selected_label])
    pool_size = db_utils.get_question_count(selected_cert["id"])

    st.sidebar.caption(f"Level: **{selected_cert['level']}** | Pass mark: **{int(selected_cert['pass_threshold']*100)}%**")
    st.sidebar.caption(f"Questions available in bank: **{pool_size}**")

    st.sidebar.divider()
    st.sidebar.markdown("**Question Source**")
    source_choice = st.sidebar.radio(
        "Question Source",
        ["📚 Use Existing Question Bank", "🤖 Generate New with AI"],
        label_visibility="collapsed",
        disabled=st.session_state.exam_stage == "in_progress",
        key="exam_question_source",
    )
    use_ai_generation = source_choice.startswith("🤖")

    num_questions = 0
    ai_provider = ai_model = ai_base_url = ai_api_key = None
    ai_exam_number = 1

    if not use_ai_generation:
        if pool_size == 0:
            st.sidebar.error("This certification has no questions yet. Add some via the Admin Dashboard, "
                              "or switch to 'Generate New with AI'.")
            return
        max_q = min(pool_size, db_utils.MAX_QUESTIONS_PER_CERT)
        if max_q <= 1:
            # Streamlit's slider requires min_value < max_value, so when only one
            # question is available we skip the slider entirely and just inform
            # the user that a single-question attempt will be generated.
            num_questions = max_q
            st.sidebar.caption("Only 1 question available — it will be used as-is.")
        else:
            num_questions = st.sidebar.slider(
                "Number of Questions",
                min_value=1,
                max_value=max_q,
                value=min(10, max_q),
                disabled=st.session_state.exam_stage == "in_progress",
            )
    else:
        st.sidebar.caption(
            "A brand-new set of questions will be generated by your selected model and "
            "saved to the local database before your exam begins."
        )
        with st.sidebar.expander("🔧 Model Provider Settings", expanded=True):
            ai_provider, ai_model, ai_base_url, ai_api_key, ai_custom_path = render_ai_provider_selector("examgen")

        room_left = max(0, db_utils.MAX_QUESTIONS_PER_CERT - pool_size)
        if room_left == 0:
            st.sidebar.warning(
                f"This certification's bank is already at the {db_utils.MAX_QUESTIONS_PER_CERT}-question "
                "cap. Switch to 'Use Existing Question Bank' or add a new certification."
            )
        else:
            sidebar_cap = min(room_left, 100)  # keep a single on-demand generation request reasonably sized
            num_questions = st.sidebar.slider(
                "Number of NEW questions to generate",
                min_value=1, max_value=sidebar_cap, value=min(10, sidebar_cap),
                disabled=st.session_state.exam_stage == "in_progress",
            )
            ai_exam_number = st.sidebar.number_input(
                "Tag as Exam Set #", min_value=1, max_value=10, value=1,
                disabled=st.session_state.exam_stage == "in_progress",
                help="Which practice-exam slot these newly generated questions are filed under.",
            )

    timed_mode = st.sidebar.toggle(
        "Timed Mode", value=False,
        disabled=st.session_state.exam_stage == "in_progress",
        help="Impose a time limit on this practice attempt, mirroring real exam conditions.",
    )
    time_limit_minutes = st.session_state.time_limit_minutes
    if timed_mode:
        time_limit_minutes = st.sidebar.number_input(
            "Time Limit (minutes)", min_value=5, max_value=240, value=90, step=5,
            disabled=st.session_state.exam_stage == "in_progress",
        )

    st.sidebar.divider()

    if st.session_state.exam_stage == "in_progress":
        st.sidebar.info("Exam in progress. Submit or reset to change configuration.")
        if st.sidebar.button("🔁 Abandon & Reset Exam", use_container_width=True):
            reset_exam()
            st.rerun()
        return

    button_label = "🤖 Generate & Start Exam" if use_ai_generation else "🚀 Start Exam"
    button_disabled = use_ai_generation and num_questions == 0
    if st.sidebar.button(button_label, type="primary", use_container_width=True, disabled=button_disabled):
        if use_ai_generation:
            if not ai_model:
                st.sidebar.error("Please select or enter a model before generating.")
                return
            try:
                progress_bar = st.sidebar.progress(0, text="Starting generation...")

                def _update_progress(done, total):
                    pct = done / total if total else 1.0
                    progress_bar.progress(min(pct, 1.0), text=f"{done} / {total} generated...")

                with st.spinner("Generating your exam with AI..."):
                    result = db_utils.generate_questions_with_ai(
                        cert=selected_cert,
                        num_questions=num_questions,
                        provider=ai_provider,
                        model=ai_model,
                        base_url=ai_base_url,
                        api_key=ai_api_key,
                        custom_path=ai_custom_path,
                        exam_number=int(ai_exam_number),
                        progress_callback=_update_progress,
                    )
                if not result["inserted_questions"]:
                    err_preview = "; ".join(result["errors"][:2]) or "No questions were returned by the model."
                    st.sidebar.error(f"Generation failed: {err_preview}")
                else:
                    if result["errors"]:
                        st.sidebar.warning(
                            f"Generated {len(result['inserted_questions'])} question(s), with "
                            f"{len(result['errors'])} issue(s) along the way."
                        )
                    start_exam(
                        selected_cert, len(result["inserted_questions"]), timed_mode, time_limit_minutes,
                        explicit_questions=result["inserted_questions"],
                    )
                    st.rerun()
            except RuntimeError as e:
                st.sidebar.error(str(e))
        else:
            start_exam(selected_cert, num_questions, timed_mode, time_limit_minutes)
            st.rerun()


# --------------------------------------------------------------------------
# UI: Exam taking screen
# --------------------------------------------------------------------------
def render_not_started():
    st.title("☁️ AWS Certification Mock Exam Engine")
    st.markdown(
        """
        Welcome! This tool lets you practice for AWS certification exams using a
        customizable question bank stored locally in SQLite.

        **How to begin:**
        1. Use the sidebar to choose your target certification.
        2. Pick how many questions you'd like to attempt.
        3. Optionally enable **Timed Mode** to simulate real exam pressure.
        4. Click **Start Exam**.

        ---
        """
    )
    summary = db_utils.get_question_counts_all()
    st.subheader("Question Bank Overview")
    cols = st.columns(len(summary)) if summary else []
    for col, row in zip(cols, summary):
        with col:
            st.metric(row["code"], f"{row['question_count']} Qs", help=row["name"])


def format_duration(total_seconds: int) -> str:
    """Format a duration in seconds as H:MM:SS (or MM:SS if under an hour)."""
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


@st.fragment(run_every=1)
def render_timer():
    """
    Live-updating timer fragment (refreshes every second without disturbing the
    exam form below it). Always shows an elapsed-time stopwatch; additionally
    shows a countdown and auto-submits the exam when Timed Mode is enabled and
    the limit is reached.
    """
    if st.session_state.exam_stage != "in_progress" or not st.session_state.exam_start_time:
        return

    elapsed = time.time() - st.session_state.exam_start_time

    if st.session_state.timed_mode:
        remaining = st.session_state.time_limit_minutes * 60 - elapsed
        if remaining <= 0:
            st.error("⏰ Time's up! Auto-submitting your exam.")
            grade_exam()
            st.rerun()
            return
        st.info(
            f"⏳ Time remaining: **{format_duration(remaining)}**  "
            f"&nbsp;|&nbsp;  ⏱️ Time elapsed: **{format_duration(elapsed)}**"
        )
    else:
        st.info(f"⏱️ Time elapsed: **{format_duration(elapsed)}**")


def render_in_progress():
    cert = st.session_state.exam_cert
    questions = st.session_state.exam_questions
    st.title(f"{cert['code']} Practice Exam")
    st.caption(cert["name"])

    render_timer()

    answered = sum(1 for v in st.session_state.user_answers.values() if v is not None)
    st.progress(answered / len(questions) if questions else 0,
                text=f"{answered} / {len(questions)} answered")

    with st.form("exam_form", clear_on_submit=False):
        option_labels = {"A": "A", "B": "B", "C": "C", "D": "D"}
        for idx, q in enumerate(questions, start=1):
            st.markdown(f"**Question {idx}.** {q['question_text']}")
            choice_display = [
                f"A. {q['option_a']}",
                f"B. {q['option_b']}",
                f"C. {q['option_c']}",
                f"D. {q['option_d']}",
            ]
            current = st.session_state.user_answers.get(q["id"])
            current_idx = "ABCD".index(current) if current in option_labels else None

            selected = st.radio(
                label=f"Select an answer for Question {idx}",
                options=choice_display,
                index=current_idx,
                key=f"radio_{q['id']}",
                label_visibility="collapsed",
            )
            # Store back into session state as soon as the widget renders on submit
            st.session_state.user_answers[q["id"]] = selected[0] if selected else None
            st.divider()

        submitted = st.form_submit_button("✅ Submit Exam", type="primary", use_container_width=True)

    if submitted:
        unanswered = [i + 1 for i, q in enumerate(questions) if st.session_state.user_answers.get(q["id"]) is None]
        if unanswered:
            st.warning(
                f"You have {len(unanswered)} unanswered question(s): "
                f"{', '.join(map(str, unanswered))}. You may still submit — "
                "unanswered questions will be marked incorrect."
            )
        grade_exam()
        st.rerun()


# --------------------------------------------------------------------------
# UI: Results & review screen
# --------------------------------------------------------------------------
def render_submitted():
    cert = st.session_state.exam_cert
    summary = st.session_state.score_summary
    questions = st.session_state.exam_questions
    answers = st.session_state.user_answers

    st.title(f"{cert['code']} — Results")

    if summary["passed"]:
        st.success(
            f"### ✅ PASS — {summary['score_pct']*100:.1f}% "
            f"(Required: {summary['threshold']*100:.0f}%)"
        )
    else:
        st.error(
            f"### ❌ FAIL — {summary['score_pct']*100:.1f}% "
            f"(Required: {summary['threshold']*100:.0f}%)"
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Correct", summary["correct"])
    c2.metric("Incorrect", summary["incorrect"])
    c3.metric("Unanswered", summary["unanswered"])
    c4.metric("Total", summary["total"])
    c5.metric("⏱️ Time Taken", format_duration(summary.get("elapsed_seconds", 0)))

    st.divider()
    st.subheader("📝 Review Mode")
    st.caption("Expand each question to see your answer versus the correct answer and the full rationale.")

    option_map = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}

    for idx, q in enumerate(questions, start=1):
        user_choice = answers.get(q["id"])
        correct_choice = q["correct_option"]
        is_correct = user_choice == correct_choice
        icon = "✅" if is_correct else ("⚪" if user_choice is None else "❌")

        with st.expander(f"{icon} Question {idx}: {q['question_text'][:90]}..."):
            st.markdown(f"**{q['question_text']}**")
            st.write("")
            for opt_key, field in option_map.items():
                text = q[field]
                prefix = f"**{opt_key}.** {text}"
                if opt_key == correct_choice:
                    st.markdown(f"🟢 {prefix}  *(Correct Answer)*")
                elif opt_key == user_choice:
                    st.markdown(f"🔴 {prefix}  *(Your Answer)*")
                else:
                    st.markdown(f"{prefix}")

            if user_choice is None:
                st.warning("You did not answer this question.")

            st.info(f"**Explanation:** {q['explanation']}")
            if q.get("domain"):
                st.caption(f"Domain: {q['domain']}")

    st.divider()
    if st.button("🔁 Start a New Exam", type="primary"):
        reset_exam()
        st.rerun()


# --------------------------------------------------------------------------
# UI: Admin Dashboard
# --------------------------------------------------------------------------
def render_admin_dashboard():
    st.title("🛠️ Admin Dashboard")
    st.caption("Manage the question bank: review counts, add questions manually, or batch-import new exams.")

    tab_overview, tab_manual, tab_batch, tab_ai, tab_new_cert = st.tabs(
        ["📊 Overview", "➕ Add Question", "📁 Batch Import", "🤖 AI Question Generator", "🏷️ Add Certification"]
    )

    certifications = db_utils.get_certifications()

    # --- Overview ---
    with tab_overview:
        st.subheader("Question Counts per Certification")
        summary = db_utils.get_question_counts_all()
        if summary:
            st.dataframe(
                summary,
                use_container_width=True,
                column_config={
                    "id": None,
                    "code": "Code",
                    "name": "Certification",
                    "level": "Level",
                    "question_count": "Question Count",
                    "cap": "Max Bank Size",
                },
                hide_index=True,
            )

            st.divider()
            st.subheader("🧹 Deduplicate Database")
            st.caption(
                f"Scans every certification's question bank and removes near-duplicate questions "
                f"(≥{int(db_utils.DUPLICATE_SIMILARITY_THRESHOLD*100)}% text-similarity to another "
                "question in the same bank), keeping the earliest-added copy of each. New questions "
                "are already checked automatically on the way in — use this to clean up anything "
                "that predates that check, or that was imported before deduplication was enabled."
            )
            if st.button("🧹 Scan & Remove Duplicates (All Certifications)"):
                with st.spinner("Scanning for near-duplicate questions..."):
                    results = db_utils.deduplicate_all_certifications()
                total_removed = sum(r["removed"] for r in results.values())
                if total_removed == 0:
                    st.success("No near-duplicate questions found. Your question bank is clean.")
                else:
                    st.success(f"Removed {total_removed} near-duplicate question(s) across all certifications.")
                    for code, r in results.items():
                        if r["removed"]:
                            st.text(f"  • {code}: removed {r['removed']}, kept {r['kept']}")
        else:
            st.info("No certifications found. Run init_db.py to seed the database.")

    # --- Manual add ---
    with tab_manual:
        st.subheader("Add a New Question")
        if not certifications:
            st.warning("No certifications available. Initialize the database first.")
        else:
            cert_labels = [f"{c['code']} — {c['name']}" for c in certifications]
            cert_lookup = {label: cert for label, cert in zip(cert_labels, certifications)}

            with st.form("add_question_form", clear_on_submit=True):
                cert_label = st.selectbox("Certification", cert_labels)
                exam_number = st.number_input(
                    "Exam Set # (1-10)", min_value=1, max_value=10, value=1,
                    help="Which full-length practice exam this question belongs to.",
                )
                question_text = st.text_area("Question Text", height=100)
                col_a, col_b = st.columns(2)
                with col_a:
                    option_a = st.text_input("Option A")
                    option_c = st.text_input("Option C")
                with col_b:
                    option_b = st.text_input("Option B")
                    option_d = st.text_input("Option D")
                correct_option = st.selectbox("Correct Option", ["A", "B", "C", "D"])
                explanation = st.text_area("Explanation / Rationale", height=100)
                domain = st.text_input("Domain (optional)")

                add_submitted = st.form_submit_button("Add Question", type="primary")

            if add_submitted:
                cert = cert_lookup[cert_label]
                try:
                    db_utils.add_question(
                        cert_id=cert["id"],
                        question_text=question_text.strip(),
                        option_a=option_a.strip(),
                        option_b=option_b.strip(),
                        option_c=option_c.strip(),
                        option_d=option_d.strip(),
                        correct_option=correct_option,
                        explanation=explanation.strip(),
                        exam_number=int(exam_number),
                        domain=domain.strip() or None,
                    )
                    st.success(f"Question added to {cert['code']} (Exam Set {exam_number}).")
                except ValueError as e:
                    st.error(f"Could not add question: {e}")

    # --- Batch import ---
    with tab_batch:
        st.subheader("Batch Import Questions (CSV or JSON)")
        st.markdown(
            """
            Upload a `.csv` or `.json` file containing multiple questions. Each record must
            include the following fields:

            `question_text, option_a, option_b, option_c, option_d, correct_option, explanation`

            Optional fields: `exam_number` (defaults to 1), `domain`.
            """
        )
        if not certifications:
            st.warning("No certifications available. Initialize the database first.")
        else:
            cert_labels = [f"{c['code']} — {c['name']}" for c in certifications]
            cert_lookup = {label: cert for label, cert in zip(cert_labels, certifications)}
            target_label = st.selectbox("Target Certification for this import", cert_labels, key="batch_cert")
            target_cert = cert_lookup[target_label]

            uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

            if uploaded_file is not None:
                try:
                    records = db_utils.parse_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
                    st.write(f"Parsed **{len(records)}** record(s) from `{uploaded_file.name}`.")
                    st.dataframe(records[:5], use_container_width=True)
                    st.caption("Preview of first 5 records.")

                    if st.button("Confirm & Import", type="primary"):
                        result = db_utils.batch_import_questions(target_cert["id"], records)
                        st.success(f"Imported {result['inserted']} question(s) into {target_cert['code']}.")
                        if result.get("duplicates_skipped"):
                            st.info(
                                f"Skipped {result['duplicates_skipped']} row(s) as near-duplicates "
                                f"(≥{int(db_utils.DUPLICATE_SIMILARITY_THRESHOLD*100)}% similar to an "
                                "existing question)."
                            )
                        other_errors = [e for e in result["errors"] if "near-duplicate" not in e]
                        if other_errors:
                            st.warning(f"{len(other_errors)} row(s) failed for other reasons:")
                            for err in other_errors:
                                st.text(f"  • {err}")
                except Exception as e:  # noqa: BLE001 - surfaced directly to the admin user
                    st.error(f"Failed to parse file: {e}")

    # --- AI question generator ---
    with tab_ai:
        st.subheader("Dynamically Generate Questions with AI")
        st.markdown(
            f"""
            Generate realistic, scenario-based practice questions on demand using
            Anthropic's cloud API, or a **local Ollama / Open WebUI** instance — no
            data leaves your machine with the local options. Each certification's
            question bank can hold up to **{db_utils.MAX_QUESTIONS_PER_CERT} questions**,
            enough for roughly 10 full-length practice exams. Generated questions are
            saved to the local SQLite database immediately.
            """
        )

        if not certifications:
            st.warning("No certifications available. Initialize the database first.")
        else:
            with st.expander("🔧 Model Provider Settings", expanded=True):
                ai_provider, ai_model, ai_base_url, ai_api_key, ai_custom_path = render_ai_provider_selector("admin_gen")

            cert_labels = [f"{c['code']} — {c['name']}" for c in certifications]
            cert_lookup = {label: cert for label, cert in zip(cert_labels, certifications)}
            gen_label = st.selectbox("Certification", cert_labels, key="ai_gen_cert")
            gen_cert = dict(cert_lookup[gen_label])

            current_count = db_utils.get_question_count(gen_cert["id"])
            room_left = max(0, db_utils.MAX_QUESTIONS_PER_CERT - current_count)
            st.caption(
                f"Current bank size: **{current_count}** / {db_utils.MAX_QUESTIONS_PER_CERT} "
                f"(**{room_left}** slot(s) remaining)"
            )

            col1, col2 = st.columns(2)
            with col1:
                exam_number = st.number_input(
                    "Exam Set # (1-10) to tag these questions with",
                    min_value=1, max_value=10, value=1, key="ai_gen_exam_number",
                )
            with col2:
                batch_size = st.selectbox(
                    "Batch size per model call", [5, 10, 15, 20], index=1,
                    help="Larger batches are faster but slightly more likely to hit output limits, "
                         "especially on smaller local models.",
                )

            if room_left == 0:
                st.info("This certification's question bank is already full.")
            else:
                num_to_generate = st.slider(
                    "Number of questions to generate",
                    min_value=1, max_value=room_left, value=min(20, room_left),
                )
                est_calls = -(-num_to_generate // batch_size)  # ceil division
                st.caption(f"This will make approximately {est_calls} model call(s).")

                if st.button("🚀 Generate Questions", type="primary"):
                    if not ai_model:
                        st.error("Please select or enter a model above (use 'Fetch Models' or type one in).")
                    else:
                        progress_bar = st.progress(0, text="Starting generation...")

                        def _update_progress(done, total):
                            pct = done / total if total else 1.0
                            progress_bar.progress(min(pct, 1.0), text=f"{done} / {total} questions generated...")

                        with st.spinner(f"Generating questions via {ai_provider}..."):
                            try:
                                result = db_utils.generate_questions_with_ai(
                                    cert=gen_cert,
                                    num_questions=num_to_generate,
                                    provider=ai_provider,
                                    model=ai_model,
                                    base_url=ai_base_url,
                                    api_key=ai_api_key,
                                    custom_path=ai_custom_path,
                                    exam_number=int(exam_number),
                                    batch_size=int(batch_size),
                                    progress_callback=_update_progress,
                                )
                                progress_bar.progress(1.0, text="Done.")
                                st.success(
                                    f"Inserted {result['inserted']} new question(s) into {gen_cert['code']} "
                                    f"across {result['batches_run']} batch(es). Saved to the local database."
                                )
                                if result["duplicates_skipped"]:
                                    st.info(f"Skipped {result['duplicates_skipped']} likely-duplicate question(s).")
                                if result["errors"]:
                                    st.warning(f"{len(result['errors'])} issue(s) encountered:")
                                    for err in result["errors"]:
                                        st.text(f"  • {err}")
                            except RuntimeError as e:
                                st.error(str(e))

    # --- Add certification ---
    with tab_new_cert:
        st.subheader("Add a New Certification")
        st.caption(
            "Not just the five built-in AWS certifications — add any certification you'd "
            "like to build a question bank for. It will immediately appear in the exam "
            "selection dropdown."
        )

        with st.form("add_certification_form", clear_on_submit=True):
            new_code = st.text_input(
                "Certification Code", placeholder="e.g. SAA-C03",
                help="Short unique identifier: letters, numbers, and hyphens only.",
            )
            new_name = st.text_input(
                "Full Certification Name",
                placeholder="e.g. AWS Certified Solutions Architect – Associate",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                new_level = st.selectbox("Level", db_utils.VALID_LEVELS)
            with col_b:
                new_threshold_pct = st.number_input(
                    "Pass Threshold (%)", min_value=1, max_value=100, value=72,
                )
            cert_submitted = st.form_submit_button("Add Certification", type="primary")

        if cert_submitted:
            try:
                new_id = db_utils.add_certification(
                    code=new_code,
                    name=new_name,
                    level=new_level,
                    pass_threshold=new_threshold_pct / 100.0,
                )
                st.success(
                    f"Added certification '{new_name}' ({new_code.strip().upper()}). "
                    "It now appears in the exam dropdown — add questions to it via the "
                    "tabs above."
                )
            except ValueError as e:
                st.error(f"Could not add certification: {e}")


# --------------------------------------------------------------------------
# Main router
# --------------------------------------------------------------------------
def main():
    view = render_sidebar()

    if view == "Admin Dashboard":
        render_admin_dashboard()
        return

    # "Take an Exam" view — routed by exam lifecycle stage
    stage = st.session_state.exam_stage
    if stage == "not_started":
        render_not_started()
    elif stage == "in_progress":
        render_in_progress()
    elif stage == "submitted":
        render_submitted()


if __name__ == "__main__":
    main()
