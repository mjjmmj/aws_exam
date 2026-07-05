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
from i18n import t, LANGUAGES, DEFAULT_LANGUAGE
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
        "language": DEFAULT_LANGUAGE,   # 'en' | 'ja' — UI display language
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


def render_language_selector():
    """Language toggle shown at the very top of the sidebar, above everything else,
    so switching languages immediately re-renders the whole app in the new language."""
    codes = list(LANGUAGES.keys())
    labels = list(LANGUAGES.values())
    current_index = codes.index(st.session_state.language) if st.session_state.language in codes else 0
    selected_label = st.sidebar.selectbox(
        t("language_label"), labels, index=current_index, key="language_selector",
    )
    selected_code = codes[labels.index(selected_label)]
    if selected_code != st.session_state.language:
        st.session_state.language = selected_code
        st.rerun()


# --------------------------------------------------------------------------
# Shared AI provider/model configuration widget
# --------------------------------------------------------------------------
def render_similarity_threshold_slider(key_prefix: str, default: float = None) -> float:
    """
    Shared slider for the duplicate-similarity threshold used when importing or
    generating questions. Returns a 0.0-1.0 fraction. 1.0 (100%) disables duplicate
    detection entirely for that operation, keeping even complete duplicates.
    """
    default_pct = int((default if default is not None else db_utils.DUPLICATE_SIMILARITY_THRESHOLD) * 100)
    pct = st.slider(
        t("similarity_threshold_label"),
        min_value=int(db_utils.MIN_SIMILARITY_THRESHOLD * 100),
        max_value=int(db_utils.MAX_SIMILARITY_THRESHOLD * 100),
        value=default_pct,
        step=1,
        key=f"{key_prefix}_similarity_threshold",
        help=t("similarity_threshold_help"),
    )
    if pct >= 100:
        st.caption(t("similarity_100_note"))
    return pct / 100.0


def render_ai_provider_selector(key_prefix: str):
    """
    Renders provider/base-URL/API-key/model widgets. Selections are persisted in
    st.session_state (shared across the Admin Dashboard and the exam-taking flow)
    so the user only has to configure their provider once per session.

    Returns (provider, model, base_url, api_key, custom_path) — model may be "" if
    none selected yet. custom_path is only meaningful for Open WebUI.
    """
    provider = st.selectbox(
        t("model_provider_label"),
        ai_providers.PROVIDERS,
        index=ai_providers.PROVIDERS.index(st.session_state.ai_provider),
        key=f"{key_prefix}_provider",
        help=t("model_provider_help"),
    )
    st.session_state.ai_provider = provider

    base_url = st.session_state.ai_base_url
    api_key = st.session_state.ai_api_key
    custom_path = st.session_state.ai_custom_path

    if ai_providers.provider_requires_base_url(provider):
        default_url = st.session_state.ai_base_url or ai_providers.default_base_url(provider)
        base_url = st.text_input(
            t("base_url_label"), value=default_url, key=f"{key_prefix}_base_url",
            help=t("base_url_help"),
        )
        st.session_state.ai_base_url = base_url

    if ai_providers.provider_requires_api_key(provider):
        placeholder = "sk-ant-..." if provider == ai_providers.PROVIDER_ANTHROPIC else "Open WebUI API key"
        env_hint = ""
        if provider == ai_providers.PROVIDER_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
            env_hint = t("api_key_env_hint")
        api_key = st.text_input(
            f"{t('api_key_label')}{env_hint}", value=st.session_state.ai_api_key, type="password",
            key=f"{key_prefix}_api_key", placeholder=placeholder,
        )
        st.session_state.ai_api_key = api_key

    if provider == ai_providers.PROVIDER_OPENWEBUI:
        with st.expander(t("advanced_custom_path")):
            st.caption(t("custom_path_caption"))
            custom_path = st.text_input(
                t("custom_path_label"),
                value=st.session_state.ai_custom_path,
                key=f"{key_prefix}_custom_path",
                placeholder="/api/chat/completions",
            )
            st.session_state.ai_custom_path = custom_path

    col_model, col_refresh = st.columns([3, 1])
    with col_refresh:
        st.write("")  # vertical alignment spacer
        fetch_clicked = st.button(t("fetch_models_button"), key=f"{key_prefix}_fetch_models")

    if fetch_clicked:
        try:
            with st.spinner(t("fetching_models_spinner")):
                models = ai_providers.list_models(provider, base_url=base_url, api_key=api_key)
            st.session_state.ai_available_models = models
            if not models:
                st.warning(t("no_models_returned"))
        except RuntimeError as e:
            st.session_state.ai_available_models = []
            st.error(str(e))

    with col_model:
        available = st.session_state.ai_available_models
        if available:
            default_idx = available.index(st.session_state.ai_model) if st.session_state.ai_model in available else 0
            model = st.selectbox(t("model_label"), available, index=default_idx, key=f"{key_prefix}_model_select")
        else:
            model = st.text_input(
                t("model_name_label"),
                value=st.session_state.ai_model,
                key=f"{key_prefix}_model_text",
                help=t("model_name_help"),
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

    If `explicit_questions` is provided (e.g. a freshly AI-generated set), those
    questions are used for this attempt — after one final duplicate-safety pass, since
    the storage-level similarity threshold used to generate them may have been
    intentionally relaxed (even to 100% / "keep complete duplicates") for the question
    bank itself. An exam must never repeat a question regardless of that setting.
    Otherwise, `num_questions` are sampled randomly from the existing question bank for
    `cert_row`, restricted to the currently selected UI language (st.session_state.language)
    so a learner using the Japanese UI is never served an English question, or vice versa.
    """
    if explicit_questions is not None:
        deduped = db_utils.deduplicate_question_list(list(explicit_questions))
        questions = deduped
    else:
        questions = db_utils.get_random_questions(
            cert_row["id"], num_questions, language=st.session_state.language
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
    render_language_selector()
    st.sidebar.title(t("app_title"))
    view_labels = [t("nav_take_exam"), t("nav_admin")]
    view_label = st.sidebar.radio(
        t("nav_label"),
        view_labels,
        index=0,
        help=t("nav_help"),
    )
    st.sidebar.divider()

    if view_label == t("nav_take_exam"):
        render_exam_config_sidebar()

    return "Admin Dashboard" if view_label == t("nav_admin") else "Take an Exam"


def render_exam_config_sidebar():
    certifications = db_utils.get_certifications()
    if not certifications:
        st.sidebar.warning(t("no_certs_found"))
        return

    st.sidebar.subheader(t("exam_config_header"))

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
        t("target_cert_label"), cert_labels, index=default_index,
        disabled=st.session_state.exam_stage == "in_progress",
    )
    selected_cert = dict(cert_lookup[selected_label])
    pool_size = db_utils.get_question_count(selected_cert["id"], language=st.session_state.language)

    st.sidebar.caption(t("level_pass_mark", level=selected_cert["level"], pct=int(selected_cert["pass_threshold"]*100)))
    st.sidebar.caption(t("questions_in_bank", count=pool_size))

    st.sidebar.divider()
    st.sidebar.markdown(t("question_source_header"))
    source_choice = st.sidebar.radio(
        t("question_source_header"),
        [t("question_source_bank"), t("question_source_ai")],
        label_visibility="collapsed",
        disabled=st.session_state.exam_stage == "in_progress",
        key="exam_question_source",
    )
    use_ai_generation = source_choice == t("question_source_ai")

    num_questions = 0
    ai_provider = ai_model = ai_base_url = ai_api_key = None
    ai_custom_path = ""
    ai_exam_number = 1
    examgen_similarity_threshold = db_utils.DUPLICATE_SIMILARITY_THRESHOLD

    if not use_ai_generation:
        if pool_size == 0:
            st.sidebar.error(t("no_questions_yet_lang", language=LANGUAGES.get(st.session_state.language, st.session_state.language)))
            return
        max_q = min(pool_size, db_utils.MAX_QUESTIONS_PER_CERT)
        if max_q <= 1:
            # Streamlit's slider requires min_value < max_value, so when only one
            # question is available we skip the slider entirely and just inform
            # the user that a single-question attempt will be generated.
            num_questions = max_q
            st.sidebar.caption(t("only_one_question"))
        else:
            num_questions = st.sidebar.slider(
                t("num_questions_label"),
                min_value=1,
                max_value=max_q,
                value=min(10, max_q),
                disabled=st.session_state.exam_stage == "in_progress",
            )
    else:
        st.sidebar.caption(t("ai_gen_intro"))
        with st.sidebar.expander(t("model_provider_settings"), expanded=True):
            ai_provider, ai_model, ai_base_url, ai_api_key, ai_custom_path = render_ai_provider_selector("examgen")
            examgen_similarity_threshold = render_similarity_threshold_slider("examgen")

        room_left = max(0, db_utils.MAX_QUESTIONS_PER_CERT - pool_size)
        if room_left == 0:
            st.sidebar.warning(t("bank_full_switch", cap=db_utils.MAX_QUESTIONS_PER_CERT))
        else:
            sidebar_cap = min(room_left, 100)  # keep a single on-demand generation request reasonably sized
            num_questions = st.sidebar.slider(
                t("num_new_questions_label"),
                min_value=1, max_value=sidebar_cap, value=min(10, sidebar_cap),
                disabled=st.session_state.exam_stage == "in_progress",
            )
            ai_exam_number = st.sidebar.number_input(
                t("tag_exam_set_label"), min_value=1, max_value=10, value=1,
                disabled=st.session_state.exam_stage == "in_progress",
                help=t("tag_exam_set_help"),
            )

    timed_mode = st.sidebar.toggle(
        t("timed_mode_label"), value=False,
        disabled=st.session_state.exam_stage == "in_progress",
        help=t("timed_mode_help"),
    )
    time_limit_minutes = st.session_state.time_limit_minutes
    if timed_mode:
        time_limit_minutes = st.sidebar.number_input(
            t("time_limit_label"), min_value=5, max_value=240, value=90, step=5,
            disabled=st.session_state.exam_stage == "in_progress",
        )

    st.sidebar.divider()

    if st.session_state.exam_stage == "in_progress":
        st.sidebar.info(t("exam_in_progress_info"))
        if st.sidebar.button(t("abandon_reset_button"), use_container_width=True):
            reset_exam()
            st.rerun()
        return

    button_label = t("generate_start_button") if use_ai_generation else t("start_exam_button")
    button_disabled = use_ai_generation and num_questions == 0
    if st.sidebar.button(button_label, type="primary", use_container_width=True, disabled=button_disabled):
        if use_ai_generation:
            if not ai_model:
                st.sidebar.error(t("select_model_first"))
                return
            try:
                progress_bar = st.sidebar.progress(0, text=t("starting_generation"))

                def _update_progress(done, total):
                    pct = done / total if total else 1.0
                    progress_bar.progress(min(pct, 1.0), text=t("progress_generated", done=done, total=total))

                with st.spinner(t("generating_exam_spinner")):
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
                        similarity_threshold=examgen_similarity_threshold,
                        language=st.session_state.language,
                    )
                if not result["inserted_questions"]:
                    err_preview = "; ".join(result["errors"][:2]) or t("no_questions_returned")
                    st.sidebar.error(t("generation_failed", detail=err_preview))
                else:
                    if result["errors"]:
                        st.sidebar.warning(
                            t("generated_with_issues", count=len(result["inserted_questions"]), issues=len(result["errors"]))
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
    st.title(t("app_title"))
    st.markdown(t("welcome_body"))
    summary = db_utils.get_question_counts_all()
    st.subheader(t("question_bank_overview"))
    cols = st.columns(len(summary)) if summary else []
    for col, row in zip(cols, summary):
        with col:
            st.metric(row["code"], t("qs_suffix", count=row["question_count"]), help=row["name"])


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
            st.error(t("time_up_autosubmit"))
            grade_exam()
            st.rerun()
            return
        st.info(t("time_remaining_elapsed", remaining=format_duration(remaining), elapsed=format_duration(elapsed)))
    else:
        st.info(t("time_elapsed_only", elapsed=format_duration(elapsed)))


def render_in_progress():
    cert = st.session_state.exam_cert
    questions = st.session_state.exam_questions
    st.title(t("practice_exam_title", code=cert["code"]))
    st.caption(cert["name"])

    render_timer()

    answered = sum(1 for v in st.session_state.user_answers.values() if v is not None)
    st.progress(answered / len(questions) if questions else 0,
                text=t("answered_progress", answered=answered, total=len(questions)))

    with st.form("exam_form", clear_on_submit=False):
        option_labels = {"A": "A", "B": "B", "C": "C", "D": "D"}
        for idx, q in enumerate(questions, start=1):
            st.markdown(t("question_number", num=idx, text=q["question_text"]))
            choice_display = [
                f"A. {q['option_a']}",
                f"B. {q['option_b']}",
                f"C. {q['option_c']}",
                f"D. {q['option_d']}",
            ]
            current = st.session_state.user_answers.get(q["id"])
            current_idx = "ABCD".index(current) if current in option_labels else None

            selected = st.radio(
                label=t("select_answer_for", num=idx),
                options=choice_display,
                index=current_idx,
                key=f"radio_{q['id']}",
                label_visibility="collapsed",
            )
            # Store back into session state as soon as the widget renders on submit
            st.session_state.user_answers[q["id"]] = selected[0] if selected else None
            st.divider()

        submitted = st.form_submit_button(t("submit_exam_button"), type="primary", use_container_width=True)

    if submitted:
        unanswered = [i + 1 for i, q in enumerate(questions) if st.session_state.user_answers.get(q["id"]) is None]
        if unanswered:
            st.warning(t("unanswered_warning", count=len(unanswered), list=", ".join(map(str, unanswered))))
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

    st.title(t("results_title", code=cert["code"]))

    if summary["passed"]:
        st.success(t("pass_banner", pct=summary["score_pct"]*100, threshold=summary["threshold"]*100))
    else:
        st.error(t("fail_banner", pct=summary["score_pct"]*100, threshold=summary["threshold"]*100))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(t("metric_correct"), summary["correct"])
    c2.metric(t("metric_incorrect"), summary["incorrect"])
    c3.metric(t("metric_unanswered"), summary["unanswered"])
    c4.metric(t("metric_total"), summary["total"])
    c5.metric(t("metric_time_taken"), format_duration(summary.get("elapsed_seconds", 0)))

    st.divider()
    st.subheader(t("review_mode_header"))
    st.caption(t("review_mode_caption"))

    option_map = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}

    for idx, q in enumerate(questions, start=1):
        user_choice = answers.get(q["id"])
        correct_choice = q["correct_option"]
        is_correct = user_choice == correct_choice
        icon = "✅" if is_correct else ("⚪" if user_choice is None else "❌")

        with st.expander(f"{icon} " + t("question_expander", num=idx, preview=q["question_text"][:90])):
            st.markdown(f"**{q['question_text']}**")
            st.write("")
            for opt_key, field in option_map.items():
                text = q[field]
                prefix = f"**{opt_key}.** {text}"
                if opt_key == correct_choice:
                    st.markdown(f"🟢 {prefix}  {t('correct_answer_tag')}")
                elif opt_key == user_choice:
                    st.markdown(f"🔴 {prefix}  {t('your_answer_tag')}")
                else:
                    st.markdown(f"{prefix}")

            if user_choice is None:
                st.warning(t("did_not_answer"))

            st.info(t("explanation_label", text=q["explanation"]))
            if q.get("domain"):
                st.caption(t("domain_label", domain=q["domain"]))

    st.divider()
    if st.button(t("start_new_exam_button"), type="primary"):
        reset_exam()
        st.rerun()


# --------------------------------------------------------------------------
# UI: Admin Dashboard
# --------------------------------------------------------------------------
def render_admin_dashboard():
    st.title(t("admin_title"))
    st.caption(t("admin_caption"))

    tab_overview, tab_manual, tab_batch, tab_ai, tab_new_cert = st.tabs(
        [t("tab_overview"), t("tab_add_question"), t("tab_batch_import"), t("tab_ai_generator"), t("tab_add_cert")]
    )

    certifications = db_utils.get_certifications()

    # --- Overview ---
    with tab_overview:
        st.subheader(t("question_counts_header"))
        summary = db_utils.get_question_counts_all()
        if summary:
            st.dataframe(
                summary,
                use_container_width=True,
                column_config={
                    "id": None,
                    "code": t("col_code"),
                    "name": t("col_certification"),
                    "level": t("col_level"),
                    "question_count": t("col_question_count"),
                    "en_count": t("col_en_count"),
                    "ja_count": t("col_ja_count"),
                    "cap": t("col_max_bank_size"),
                },
                hide_index=True,
            )

            st.divider()
            st.subheader(t("dedup_header"))
            st.caption(t("dedup_caption", pct=int(db_utils.DUPLICATE_SIMILARITY_THRESHOLD*100)))
            dedup_similarity_threshold = render_similarity_threshold_slider("cleanup_dedup")
            if st.button(t("dedup_button")):
                with st.spinner(t("dedup_scanning")):
                    results = db_utils.deduplicate_all_certifications(threshold=dedup_similarity_threshold)
                total_removed = sum(r["removed"] for r in results.values())
                if total_removed == 0:
                    st.success(t("dedup_none_found"))
                else:
                    st.success(t("dedup_removed_summary", count=total_removed))
                    for code, r in results.items():
                        if r["removed"]:
                            st.text(t("dedup_removed_line", code=code, removed=r["removed"], kept=r["kept"]))
        else:
            st.info(t("no_certs_init_db"))

    # --- Manual add ---
    with tab_manual:
        st.subheader(t("add_question_header"))
        if not certifications:
            st.warning(t("no_certs_init_first"))
        else:
            cert_labels = [f"{c['code']} — {c['name']}" for c in certifications]
            cert_lookup = {label: cert for label, cert in zip(cert_labels, certifications)}

            with st.form("add_question_form", clear_on_submit=True):
                cert_label = st.selectbox(t("certification_label"), cert_labels)
                lang_codes = list(LANGUAGES.keys())
                lang_labels = list(LANGUAGES.values())
                default_lang_idx = lang_codes.index(st.session_state.language) if st.session_state.language in lang_codes else 0
                question_lang_label = st.selectbox(
                    t("question_language_label"), lang_labels, index=default_lang_idx,
                    help=t("question_language_help"),
                )
                question_language = lang_codes[lang_labels.index(question_lang_label)]
                exam_number = st.number_input(
                    t("exam_set_label"), min_value=1, max_value=10, value=1,
                    help=t("exam_set_help"),
                )
                question_text = st.text_area(t("question_text_label"), height=100)
                col_a, col_b = st.columns(2)
                with col_a:
                    option_a = st.text_input(t("option_a_label"))
                    option_c = st.text_input(t("option_c_label"))
                with col_b:
                    option_b = st.text_input(t("option_b_label"))
                    option_d = st.text_input(t("option_d_label"))
                correct_option = st.selectbox(t("correct_option_label"), ["A", "B", "C", "D"])
                explanation = st.text_area(t("explanation_field_label"), height=100)
                domain = st.text_input(t("domain_optional_label"))

                add_submitted = st.form_submit_button(t("add_question_button"), type="primary")

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
                        language=question_language,
                    )
                    st.success(t("question_added_success", code=cert["code"], exam_number=exam_number))
                except ValueError as e:
                    st.error(t("could_not_add_question", error=str(e)))

    # --- Batch import ---
    with tab_batch:
        st.subheader(t("batch_import_header"))
        st.markdown(t("batch_import_instructions"))
        if not certifications:
            st.warning(t("no_certs_init_first"))
        else:
            cert_labels = [f"{c['code']} — {c['name']}" for c in certifications]
            cert_lookup = {label: cert for label, cert in zip(cert_labels, certifications)}
            target_label = st.selectbox(t("target_cert_import_label"), cert_labels, key="batch_cert")
            target_cert = cert_lookup[target_label]

            lang_codes = list(LANGUAGES.keys())
            lang_labels = list(LANGUAGES.values())
            default_lang_idx = lang_codes.index(st.session_state.language) if st.session_state.language in lang_codes else 0
            batch_lang_label = st.selectbox(
                t("default_language_label"), lang_labels, index=default_lang_idx,
                help=t("default_language_help"),
            )
            batch_default_language = lang_codes[lang_labels.index(batch_lang_label)]

            batch_similarity_threshold = render_similarity_threshold_slider("batch_import")

            uploaded_file = st.file_uploader(t("upload_file_label"), type=["csv", "json"])

            if uploaded_file is not None:
                try:
                    records = db_utils.parse_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
                    st.write(t("parsed_records", count=len(records), filename=uploaded_file.name))
                    st.dataframe(records[:5], use_container_width=True)
                    st.caption(t("preview_first_5"))

                    if st.button(t("confirm_import_button"), type="primary"):
                        result = db_utils.batch_import_questions(
                            target_cert["id"], records, similarity_threshold=batch_similarity_threshold,
                            default_language=batch_default_language,
                        )
                        st.success(t("imported_success", count=result["inserted"], code=target_cert["code"]))
                        if result.get("duplicates_skipped"):
                            st.info(
                                t("skipped_duplicates_info",
                                  count=result["duplicates_skipped"],
                                  pct=int(batch_similarity_threshold*100))
                            )
                        other_errors = [e for e in result["errors"] if "near-duplicate" not in e]
                        if other_errors:
                            st.warning(t("rows_failed_other", count=len(other_errors)))
                            for err in other_errors:
                                st.text(f"  • {err}")
                except Exception as e:  # noqa: BLE001 - surfaced directly to the admin user
                    st.error(t("failed_to_parse", error=str(e)))

    # --- AI question generator ---
    with tab_ai:
        st.subheader(t("ai_generator_subheader"))
        st.markdown(t("ai_generator_intro", cap=db_utils.MAX_QUESTIONS_PER_CERT))

        if not certifications:
            st.warning(t("no_certs_init_first"))
        else:
            with st.expander(t("model_provider_settings"), expanded=True):
                ai_provider, ai_model, ai_base_url, ai_api_key, ai_custom_path = render_ai_provider_selector("admin_gen")

            cert_labels = [f"{c['code']} — {c['name']}" for c in certifications]
            cert_lookup = {label: cert for label, cert in zip(cert_labels, certifications)}
            gen_label = st.selectbox(t("certification_label"), cert_labels, key="ai_gen_cert")
            gen_cert = dict(cert_lookup[gen_label])

            lang_codes = list(LANGUAGES.keys())
            lang_labels = list(LANGUAGES.values())
            default_lang_idx = lang_codes.index(st.session_state.language) if st.session_state.language in lang_codes else 0
            gen_lang_label = st.selectbox(
                t("question_language_label"), lang_labels, index=default_lang_idx,
                help=t("question_language_help"), key="ai_gen_language",
            )
            gen_language = lang_codes[lang_labels.index(gen_lang_label)]

            current_count = db_utils.get_question_count(gen_cert["id"], language=gen_language)
            room_left = max(0, db_utils.MAX_QUESTIONS_PER_CERT - current_count)
            st.caption(t("bank_size_caption", current=current_count, cap=db_utils.MAX_QUESTIONS_PER_CERT, room=room_left))

            col1, col2 = st.columns(2)
            with col1:
                exam_number = st.number_input(
                    t("exam_set_tag_label"),
                    min_value=1, max_value=10, value=1, key="ai_gen_exam_number",
                )
            with col2:
                batch_size = st.selectbox(
                    t("batch_size_label"), [5, 10, 15, 20], index=1,
                    help=t("batch_size_help"),
                )

            gen_similarity_threshold = render_similarity_threshold_slider("admin_gen")

            if room_left == 0:
                st.info(t("bank_already_full"))
            else:
                num_to_generate = st.slider(
                    t("num_to_generate_label"),
                    min_value=1, max_value=room_left, value=min(20, room_left),
                )
                est_calls = -(-num_to_generate // batch_size)  # ceil division
                st.caption(t("approx_calls_caption", count=est_calls))

                if st.button(t("generate_button"), type="primary"):
                    if not ai_model:
                        st.error(t("select_model_first"))
                    else:
                        progress_bar = st.progress(0, text=t("starting_generation"))

                        def _update_progress(done, total):
                            pct = done / total if total else 1.0
                            progress_bar.progress(min(pct, 1.0), text=t("progress_generated", done=done, total=total))

                        with st.spinner(t("generating_via_spinner", provider=ai_provider)):
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
                                    similarity_threshold=gen_similarity_threshold,
                                    language=gen_language,
                                )
                                progress_bar.progress(1.0, text=t("done_progress"))
                                st.success(
                                    t("inserted_success", inserted=result["inserted"], code=gen_cert["code"],
                                      batches=result["batches_run"])
                                )
                                if result["duplicates_skipped"]:
                                    st.info(t("skipped_dup_info", count=result["duplicates_skipped"]))
                                if result["errors"]:
                                    st.warning(t("issues_encountered", count=len(result["errors"])))
                                    for err in result["errors"]:
                                        st.text(f"  • {err}")
                            except RuntimeError as e:
                                st.error(str(e))

    # --- Add certification ---
    with tab_new_cert:
        st.subheader(t("add_cert_header"))
        st.caption(t("add_cert_caption"))

        with st.form("add_certification_form", clear_on_submit=True):
            new_code = st.text_input(
                t("cert_code_label"), placeholder="e.g. SAA-C03",
                help=t("cert_code_help"),
            )
            new_name = st.text_input(
                t("cert_name_label"),
                placeholder="e.g. AWS Certified Solutions Architect – Associate",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                new_level = st.selectbox(t("level_label"), db_utils.VALID_LEVELS)
            with col_b:
                new_threshold_pct = st.number_input(
                    t("pass_threshold_label"), min_value=1, max_value=100, value=72,
                )
            cert_submitted = st.form_submit_button(t("add_cert_button"), type="primary")

        if cert_submitted:
            try:
                new_id = db_utils.add_certification(
                    code=new_code,
                    name=new_name,
                    level=new_level,
                    pass_threshold=new_threshold_pct / 100.0,
                )
                st.success(t("cert_added_success", name=new_name, code=new_code.strip().upper()))
            except ValueError as e:
                st.error(t("could_not_add_cert", error=str(e)))


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
