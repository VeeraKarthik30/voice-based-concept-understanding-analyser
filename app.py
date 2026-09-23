from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from authentication.auth import AuthService
from authentication.user_session import SessionManager
from database.db import DatabaseManager
from modules.audio_analysis import AudioAnalysisError, analyse_audio_file
from modules.scoring_engine import ScoringEngine
from modules.semantic_analysis import SemanticAnalysisError, analyse_concept_understanding
from modules.transcription import TranscriptionError, transcribe_audio
from modules.waveform import build_metric_chart, build_waveform_figure
from utils.config import AppConfig, ensure_runtime_directories, get_app_config
from utils.helpers import (
    AnalysisPayload,
    coerce_float,
    current_timestamp,
    format_items_for_display,
    safe_slug,
    sanitize_html_text,
    save_uploaded_file,
    validate_audio_upload,
)
from utils.pdf_generator import build_pdf_report


st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PUBLIC_PAGES = {"landing", "login", "signup"}
APP_PAGES = {
    "dashboard": "Dashboard",
    "upload": "Upload Audio",
    "results": "Analysis Results",
    "history": "History",
    "profile": "Profile",
}


def _supports_parameter(func: Any, parameter_name: str) -> bool:
    try:
        return parameter_name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def full_width_kwargs(func: Any, *, image_fallback: bool = False) -> dict[str, Any]:
    if _supports_parameter(func, "use_container_width"):
        return {"use_container_width": True}
    if image_fallback and _supports_parameter(func, "use_column_width"):
        return {"use_column_width": True}
    if _supports_parameter(func, "width"):
        return {"width": "stretch"}
    return {}


def load_css(config: AppConfig, theme: str) -> None:
    theme_key = (theme or "dark").lower()
    candidates = [
        config.assets_dir / f"styles_{theme_key}.css",
        config.assets_dir / "styles.css",
    ]
    css_path = next((path for path in candidates if path.exists()), None)
    if css_path:
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_database() -> DatabaseManager:
    config = get_app_config()
    ensure_runtime_directories(config)
    return DatabaseManager(config.database_path)


@st.cache_resource(show_spinner=False)
def get_auth_service() -> AuthService:
    return AuthService(get_database())


def navigate(page: str) -> None:
    st.session_state["page"] = page


def show_notice() -> None:
    notice = st.session_state.pop("auth_notice", None)
    if notice:
        st.success(notice)


def render_stat_card(title: str, value: float, help_text: str, tone: str = "default") -> None:
    safe_title = sanitize_html_text(title)
    safe_help_text = sanitize_html_text(help_text)
    st.markdown(
        f"""
        <div class="metric-card tone-{tone}">
            <div class="metric-title">{safe_title}</div>
            <div class="metric-value">{value:.1f}</div>
            <div class="metric-subtitle">{safe_help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, body: str) -> None:
    safe_kicker = sanitize_html_text(kicker)
    safe_title = sanitize_html_text(title)
    safe_body = sanitize_html_text(body)
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-kicker">{safe_kicker}</div>
            <h2>{safe_title}</h2>
            <p>{safe_body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shell_header(title: str, subtitle: str) -> None:
    safe_title = sanitize_html_text(title)
    safe_subtitle = sanitize_html_text(subtitle)
    st.markdown(
        f"""
        <div class="page-shell">
            <div class="page-title">{safe_title}</div>
            <div class="page-subtitle">{safe_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_nav(current_page: str) -> None:
    nav_columns = st.columns(len(APP_PAGES))
    for index, (page_key, page_label) in enumerate(APP_PAGES.items()):
        button_type = "primary" if page_key == current_page else "secondary"
        with nav_columns[index]:
            if st.button(
                page_label,
                key=f"workspace_nav_{current_page}_{page_key}",
                type=button_type,
                **full_width_kwargs(st.button),
            ):
                if page_key != current_page:
                    navigate(page_key)
                    st.rerun()


def render_theme_table(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return

    headers = "".join(f"<th>{sanitize_html_text(column)}</th>" for column in dataframe.columns)
    rows = []
    for _, row in dataframe.iterrows():
        cells = "".join(f"<td>{sanitize_html_text(value)}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    st.markdown(
        f"""
        <div class="theme-table-wrap">
            <table class="theme-table">
                <thead>
                    <tr>{headers}</tr>
                </thead>
                <tbody>
                    {body}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_history_dataframe(records: list[Any]) -> pd.DataFrame:
    rows = []
    for record in records:
        scores = json.loads(record["scores_json"])
        rows.append(
            {
                "Created At": record["created_at"],
                "Concept": record["concept_name"] or "Untitled concept",
                "Filename": record["original_filename"],
                "Final": coerce_float(scores.get("final")),
                "Understanding": coerce_float(scores.get("understanding")),
                "Fluency": coerce_float(scores.get("fluency")),
                "Communication": coerce_float(scores.get("communication")),
                "Accuracy": coerce_float(scores.get("accuracy")),
                "Report": record["report_path"],
                "Transcription": record["transcription"],
            }
        )
    return pd.DataFrame(rows)


def build_trend_figure(history_df: pd.DataFrame, theme: str = "dark") -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    trend = history_df.iloc[::-1].reset_index(drop=True)
    axis.plot(trend.index + 1, trend["Final"], color="#7c5cff", linewidth=2.4, marker="o")
    axis.fill_between(trend.index + 1, trend["Final"], color="#7c5cff", alpha=0.12)
    theme_key = (theme or "dark").lower()
    if theme_key == "light":
        bg = "#ffffff"
        fg = "#101828"
        grid = "#d0d5dd"
        spine = "#d0d5dd"
    else:
        bg = "#12192b"
        fg = "#f6f8ff"
        grid = "#293250"
        spine = "#293250"

    axis.set_title("Final score trend", color=fg)
    axis.set_xlabel("Attempt")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.18, color=grid)
    figure.patch.set_facecolor(bg)
    axis.set_facecolor(bg)
    axis.tick_params(colors=fg)
    axis.xaxis.label.set_color(fg)
    axis.yaxis.label.set_color(fg)
    for axis_spine in axis.spines.values():
        axis_spine.set_color(spine)
    return figure


def ensure_latest_analysis_theme_assets(result: dict[str, Any] | None, config: AppConfig) -> dict[str, Any] | None:
    if not result:
        return result

    current_theme = st.session_state.get("theme", "dark")
    analysis_stamp = result.get("analysis_stamp")
    analysis_slug = result.get("analysis_slug")
    audio_raw = result.get("audio_raw") or result.get("audio")
    scoring = result.get("scoring")
    waveform_path_raw = result.get("waveform_path")
    metric_chart_path_raw = result.get("metric_chart_path")

    waveform_exists = bool(waveform_path_raw and Path(waveform_path_raw).exists())
    metric_chart_exists = bool(metric_chart_path_raw and Path(metric_chart_path_raw).exists())
    if result.get("theme") == current_theme and waveform_exists and metric_chart_exists:
        return result

    if not analysis_stamp or not analysis_slug or not audio_raw or not scoring:
        return result

    waveform_path = config.reports_dir / f"waveform_{analysis_stamp}_{analysis_slug}_{current_theme}.png"
    metric_chart_path = config.reports_dir / f"metrics_{analysis_stamp}_{analysis_slug}_{current_theme}.png"
    build_waveform_figure(audio_raw, str(waveform_path), theme=current_theme)
    build_metric_chart(scoring["scores"], str(metric_chart_path), theme=current_theme)

    report_path = result.get("report_path")
    user_details = result.get("report_user_details")
    transcription_payload = result.get("transcription", {})
    if isinstance(transcription_payload, dict):
        transcription_text = transcription_payload.get("text", "")
    else:
        transcription_text = str(transcription_payload or "")
    semantic = result.get("semantic", {})
    audio_metrics = result.get("audio_metrics") or {
        key: value for key, value in audio_raw.items() if key != "signal"
    }
    if report_path and user_details and transcription_text and semantic:
        build_pdf_report(
            output_path=str(report_path),
            transcription_text=transcription_text,
            scores=scoring["scores"],
            strengths=scoring["strengths"],
            weaknesses=scoring["weaknesses"],
            suggestions=scoring["suggestions"],
            audio_metrics=audio_metrics,
            semantic_metrics=semantic,
            chart_paths=[str(waveform_path), str(metric_chart_path)],
            user_details=user_details,
        )

    refreshed_result = {
        **result,
        "theme": current_theme,
        "waveform_path": str(waveform_path),
        "metric_chart_path": str(metric_chart_path),
        "audio_metrics": audio_metrics,
    }
    st.session_state["latest_analysis"] = refreshed_result
    return refreshed_result


def render_history_download(report_path: str, label: str) -> None:
    report_file = Path(report_path)
    if report_file.exists():
        with open(report_file, "rb") as file_obj:
            st.download_button(
                label=label,
                data=file_obj.read(),
                file_name=report_file.name,
                mime="application/pdf",
                **full_width_kwargs(st.download_button),
            )
    else:
        st.caption("Report file not found yet. Generate a new report to download it again.")


def render_public_topbar() -> None:
    st.markdown("<div class='topbar-space'></div>", unsafe_allow_html=True)
    left, middle, right = st.columns([1.6, 4.6, 2.4], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class='brand-mark'>
                <div class='brand-short'>VBCUA</div>
                <div class='brand-full'>Voice-Based Concept Understanding Analyser</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            "<div class='topbar-links'>Speech Intelligence <span>•</span> Semantic Scoring <span>•</span> PDF Reports</div>",
            unsafe_allow_html=True,
        )
    with right:
        theme_value = st.session_state.get("theme", "dark")
        light_mode = st.toggle("Light mode", value=theme_value == "light", key="public_theme_toggle")
        desired_theme = "light" if light_mode else "dark"
        if desired_theme != theme_value:
            st.session_state["theme"] = desired_theme
            st.rerun()

        action_left, action_right = st.columns(2)
        with action_left:
            if st.button("Log in", key="topbar_login", **full_width_kwargs(st.button)):
                navigate("login")
                st.rerun()
        with action_right:
            if st.button("Get started", key="topbar_signup", type="primary", **full_width_kwargs(st.button)):
                navigate("signup")
                st.rerun()


def render_app_sidebar(user: dict[str, Any]) -> str:
    safe_name = sanitize_html_text(user["full_name"])
    safe_email = sanitize_html_text(user["email"])
    current_page = st.session_state.get("page", "dashboard")
    if current_page not in APP_PAGES:
        current_page = "dashboard"
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-badge">Premium Workspace</div>
                <h3>Voice-Based Concept Understanding Analyser</h3>
                <p>AI voice analytics for concept mastery and speaking performance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="user-chip">
                <div class="user-chip-title">{safe_name}</div>
                <div class="user-chip-subtitle">{safe_email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        theme_value = st.session_state.get("theme", "dark")
        light_mode = st.toggle("Light mode", value=theme_value == "light")
        desired_theme = "light" if light_mode else "dark"
        if desired_theme != theme_value:
            st.session_state["theme"] = desired_theme
            st.rerun()

        st.markdown("### Workspace")
        selected_page = current_page
        for page_key, page_label in APP_PAGES.items():
            button_type = "primary" if page_key == current_page else "secondary"
            if st.button(
                page_label,
                key=f"sidebar_nav_{page_key}",
                type=button_type,
                **full_width_kwargs(st.button),
            ):
                selected_page = page_key

        st.markdown("<div class='sidebar-footer'>Built for premium oral assessment workflows.</div>", unsafe_allow_html=True)
        if st.button("Logout", **full_width_kwargs(st.button)):
            SessionManager.logout_user()
            st.rerun()
    return selected_page


def landing_page() -> None:
    render_public_topbar()
    show_notice()

    hero_left, hero_right = st.columns([1.25, 1], vertical_alignment="center")
    with hero_left:
        st.markdown(
            """
            <div class="hero-card premium-hero">
                <div class="hero-pill">Voice intelligence for concept mastery</div>
                <h1>Voice-Based Concept Understanding Analyser</h1>
                <p>
                    Analyse how well a learner explains a concept through voice using Whisper transcription,
                    Sentence-BERT semantic comparison, expected concept matching, and concept understanding scoring.
                </p>
                <div class="hero-feature-row">
                    <span>Concept-based evaluation</span>
                    <span>Whisper transcription</span>
                    <span>Sentence-BERT comparison</span>
                    <span>Concept understanding score</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        action_left, action_right = st.columns(2)
        with action_left:
            if st.button("Open login page", key="landing_login", **full_width_kwargs(st.button)):
                navigate("login")
                st.rerun()
        with action_right:
            if st.button("Create account", key="landing_signup", type="primary", **full_width_kwargs(st.button)):
                navigate("signup")
                st.rerun()

    with hero_right:
        st.markdown(
            """
            <div class="spotlight-card flow-card">
                <div class="spotlight-kicker">System flow</div>
                <p>End-to-end voice analysis pipeline for concept understanding.</p>
                <div class="flow-sequence">
                    <div class="flow-step">User selects Concept</div>
                    <div class="flow-arrow">↓</div>
                    <div class="flow-step">System loads Reference Answer + Expected Concepts</div>
                    <div class="flow-arrow">↓</div>
                    <div class="flow-step">User uploads voice explanation</div>
                    <div class="flow-arrow">↓</div>
                    <div class="flow-step">Whisper converts voice → text</div>
                    <div class="flow-arrow">↓</div>
                    <div class="flow-step">Sentence-BERT compares: (User Explanation vs Expected Knowledge)</div>
                    <div class="flow-arrow">↓</div>
                    <div class="flow-step final-step">Concept Understanding Score</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_header(
        "Why it feels premium",
        "A real product structure instead of a single demo page",
        "Separate landing, login, signup, dashboard, upload, results, history, and profile flows create a cleaner user journey.",
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        render_stat_card("Concept insight", 94.0, "Semantic scoring against a reference answer", "violet")
    with metric_col2:
        render_stat_card("Delivery quality", 89.0, "Pause, filler, and clarity analysis", "cyan")
    with metric_col3:
        render_stat_card("Report readiness", 100.0, "Downloadable PDF evidence after each run", "green")

    feature_col1, feature_col2 = st.columns([1.3, 1], vertical_alignment="top")
    with feature_col1:
        st.markdown(
            """
            <div class="glass-panel">
                <h3>Product highlights</h3>
                <div class="feature-grid">
                    <div class="feature-item">
                        <h4>Separate auth experience</h4>
                        <p>Landing, login, and signup are split into focused, polished screens.</p>
                    </div>
                    <div class="feature-item">
                        <h4>Professional analytics</h4>
                        <p>Scorecards, history tracking, and visual summaries feel closer to a SaaS dashboard.</p>
                    </div>
                    <div class="feature-item">
                        <h4>AI evaluation pipeline</h4>
                        <p>Whisper, SentenceTransformers, and librosa work together in one modular flow.</p>
                    </div>
                    <div class="feature-item">
                        <h4>Ready for reports</h4>
                        <p>Each analysis can be exported as a polished PDF for assessment records.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feature_col2:
        steps = pd.DataFrame(
            [
                {"Stage": "01", "Summary": "Upload spoken answer and context"},
                {"Stage": "02", "Summary": "Transcribe using local Whisper"},
                {"Stage": "03", "Summary": "Evaluate concept understanding and fluency"},
                {"Stage": "04", "Summary": "Export PDF and track performance history"},
            ]
        )
        render_theme_table(steps)


def auth_page_frame(title: str, subtitle: str) -> tuple[Any, Any, Any]:
    render_public_topbar()
    show_notice()
    left, center, right = st.columns([1, 1.1, 1], vertical_alignment="center")
    with center:
        safe_title = sanitize_html_text(title)
        safe_subtitle = sanitize_html_text(subtitle)
        st.markdown(
            f"""
            <div class="auth-shell">
                <div class="auth-kicker">Secure account access</div>
                <h1>{safe_title}</h1>
                <p>{safe_subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return left, center, right


def login_page(auth_service: AuthService) -> None:
    _, center, _ = auth_page_frame(
        "Welcome back",
        "Access your voice analytics workspace, review past evaluations, and generate new reports.",
    )
    with center:
        default_email = st.session_state.get("login_email", "")
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email address", value=default_email, placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Log in", type="primary", **full_width_kwargs(st.form_submit_button))
        if submitted:
            success, message, user = auth_service.login(email, password)
            if success and user:
                SessionManager.login_user(user)
                st.session_state["auth_notice"] = "Login successful. Your premium workspace is ready."
                st.rerun()
            st.error(message)

        alt_left, alt_right = st.columns(2)
        with alt_left:
            if st.button("Back to home", **full_width_kwargs(st.button)):
                navigate("landing")
                st.rerun()
        with alt_right:
            if st.button("Go to signup", **full_width_kwargs(st.button)):
                navigate("signup")
                st.rerun()


def signup_page(auth_service: AuthService) -> None:
    _, center, _ = auth_page_frame(
        "Create your workspace",
        "Start with a dedicated account for upload history, scoring dashboards, and professional report generation.",
    )
    with center:
        with st.form("signup_form", clear_on_submit=False):
            full_name = st.text_input("Full name", placeholder="Enter your full name")
            email = st.text_input("Email address", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Use at least 8 characters")
            confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password")
            submitted = st.form_submit_button("Create account", type="primary", **full_width_kwargs(st.form_submit_button))
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = auth_service.signup(full_name, email, password)
                if success:
                    st.session_state["login_email"] = email.strip().lower()
                    st.session_state["auth_notice"] = "Account created successfully. Please log in."
                    navigate("login")
                    st.rerun()
                st.error(message)

        alt_left, alt_right = st.columns(2)
        with alt_left:
            if st.button("Back to home", key="signup_home", **full_width_kwargs(st.button)):
                navigate("landing")
                st.rerun()
        with alt_right:
            if st.button("Go to login", key="signup_login", **full_width_kwargs(st.button)):
                navigate("login")
                st.rerun()


def analysis_inputs() -> tuple[str, str, str]:
    concept_name = st.text_input("Concept name", placeholder="e.g., Neural Networks, Photosynthesis, Newton's Laws")
    prompt_left, prompt_right = st.columns(2)
    with prompt_left:
        reference_answer = st.text_area(
            "Reference answer",
            height=210,
            placeholder="Paste the ideal answer, marking scheme, or rubric-aligned explanation.",
        )
    with prompt_right:
        expected_concepts = st.text_area(
            "Expected concepts (optional)",
            height=210,
            placeholder="Add concepts or learning outcomes separated by commas.",
        )
    return concept_name.strip(), reference_answer.strip(), expected_concepts.strip()


def run_analysis(
    uploaded_file: Any,
    concept_name: str,
    reference_answer: str,
    expected_concepts: str,
    db: DatabaseManager,
    user: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    user_id = int(user["id"])
    is_valid_upload, validation_message = validate_audio_upload(uploaded_file)
    if not is_valid_upload:
        raise ValueError(validation_message)

    audio_path = save_uploaded_file(uploaded_file, config.uploads_dir)

    transcription_result = transcribe_audio(str(audio_path))
    audio_result = analyse_audio_file(str(audio_path), transcription_result["text"])
    semantic_result = analyse_concept_understanding(
        learner_answer=transcription_result["text"],
        reference_answer=reference_answer,
        expected_concepts=expected_concepts,
    )
    scoring_result = ScoringEngine().generate_scorecard(
        semantic_result=semantic_result,
        audio_result=audio_result,
        transcription_text=transcription_result["text"],
    )

    analysis_stamp = current_timestamp("%Y%m%d_%H%M%S")
    slug = safe_slug(concept_name or Path(uploaded_file.name).stem)
    theme = st.session_state.get("theme", "dark")
    waveform_path = config.reports_dir / f"waveform_{analysis_stamp}_{slug}_{theme}.png"
    metric_chart_path = config.reports_dir / f"metrics_{analysis_stamp}_{slug}_{theme}.png"
    build_waveform_figure(audio_result, str(waveform_path), theme=theme)
    build_metric_chart(scoring_result["scores"], str(metric_chart_path), theme=theme)
    persisted_audio_result = {key: value for key, value in audio_result.items() if key != "signal"}

    report_path = config.reports_dir / f"report_{analysis_stamp}_{slug}.pdf"
    report_user_details = {
        "full_name": user["full_name"],
        "email": user["email"],
        "generated_at": current_timestamp(),
        "audio_filename": uploaded_file.name,
        "concept_name": concept_name,
    }
    build_pdf_report(
        output_path=str(report_path),
        transcription_text=transcription_result["text"],
        scores=scoring_result["scores"],
        strengths=scoring_result["strengths"],
        weaknesses=scoring_result["weaknesses"],
        suggestions=scoring_result["suggestions"],
        audio_metrics=persisted_audio_result,
        semantic_metrics=semantic_result,
        chart_paths=[str(waveform_path), str(metric_chart_path)],
        user_details=report_user_details,
    )

    payload = AnalysisPayload(
        user_id=user_id,
        concept_name=concept_name or "Untitled concept",
        original_filename=uploaded_file.name,
        stored_audio_path=str(audio_path),
        transcription=transcription_result["text"],
        scores=scoring_result["scores"],
        report_path=str(report_path),
    )
    analysis_id = db.save_analysis(payload)

    return {
        "analysis_id": analysis_id,
        "audio_path": str(audio_path),
        "original_filename": uploaded_file.name,
        "concept_name": concept_name,
        "report_path": str(report_path),
        "waveform_path": str(waveform_path),
        "metric_chart_path": str(metric_chart_path),
        "analysis_stamp": analysis_stamp,
        "analysis_slug": slug,
        "theme": theme,
        "audio": persisted_audio_result,
        "audio_metrics": persisted_audio_result,
        "audio_raw": audio_result,
        "transcription": transcription_result,
        "semantic": semantic_result,
        "scoring": scoring_result,
        "report_user_details": report_user_details,
        "evaluated_at": current_timestamp(),
    }


def dashboard_page(db: DatabaseManager, user_id: int) -> None:
    render_shell_header("Dashboard", "Track performance, review recent reports, and launch a new voice assessment.")
    records = db.fetch_user_history(user_id)
    history_df = build_history_dataframe(records) if records else pd.DataFrame()
    summary = db.fetch_user_summary(user_id)

    latest_row = history_df.iloc[0] if not history_df.empty else None
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card("Analyses", float(summary["analysis_count"]), "Completed evaluations", "violet")
    with col2:
        render_stat_card("Average final", summary["average_final_score"], "Average overall performance", "cyan")
    with col3:
        render_stat_card(
            "Best final",
            float(history_df["Final"].max()) if not history_df.empty else 0.0,
            "Highest recorded performance",
            "green",
        )
    with col4:
        render_stat_card(
            "Latest result",
            float(latest_row["Final"]) if latest_row is not None else 0.0,
            "Most recent analysis score",
            "amber",
        )

    main_left, main_right = st.columns([1.3, 1], vertical_alignment="top")
    with main_left:
        st.markdown("<div class='panel-title'>Performance overview</div>", unsafe_allow_html=True)
        if history_df.empty:
            st.markdown(
                """
                <div class="empty-state">
                    <h3>No analyses yet</h3>
                    <p>Upload your first spoken answer to populate the dashboard, metrics, and reports.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Start first analysis", key="dash_start_first", type="primary"):
                navigate("upload")
                st.rerun()
        else:
            st.pyplot(
                build_trend_figure(history_df, theme=st.session_state.get("theme", "dark")),
                **full_width_kwargs(st.pyplot),
            )
            recent = history_df[["Created At", "Concept", "Final", "Understanding", "Fluency", "Communication"]].head(5)
            render_theme_table(recent)

    with main_right:
        st.markdown(
            """
            <div class="highlight-panel">
                <div class="panel-kicker">Quick actions</div>
                <h3>Keep the workflow moving</h3>
                <p>Jump straight into a new analysis, revisit the latest result, or open your history archive.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        quick1, quick2 = st.columns(2)
        with quick1:
            if st.button("New analysis", key="dashboard_to_upload", **full_width_kwargs(st.button)):
                navigate("upload")
                st.rerun()
        with quick2:
            if st.button("Open history", key="dashboard_to_history", **full_width_kwargs(st.button)):
                navigate("history")
                st.rerun()
        if latest_row is not None:
            safe_latest_concept = sanitize_html_text(latest_row["Concept"])
            safe_latest_filename = sanitize_html_text(latest_row["Filename"])
            safe_latest_created = sanitize_html_text(latest_row["Created At"])
            st.markdown(
                f"""
                <div class="glass-panel compact-panel">
                    <h4>Latest submission</h4>
                    <p><strong>{safe_latest_concept}</strong></p>
                    <p class="metric-subtitle">{safe_latest_filename}</p>
                    <p>Final score: {latest_row["Final"]:.1f}</p>
                    <p>Generated: {safe_latest_created}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_history_download(str(latest_row["Report"]), "Download latest report")


def upload_page(db: DatabaseManager, user: dict[str, Any], config: AppConfig) -> None:
    render_shell_header("Upload and analyse", "Submit a spoken answer, attach the ideal answer, and generate a polished AI evaluation.")

    panel_left, panel_right = st.columns([1.25, 1], vertical_alignment="top")
    with panel_left:
        st.markdown(
            """
            <div class="glass-panel">
                <h3>Assessment input</h3>
                <p>Provide the expected answer and concept targets so the semantic model can measure understanding accurately.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        concept_name, reference_answer, expected_concepts = analysis_inputs()
    with panel_right:
        st.markdown(
            """
            <div class="glass-panel">
                <h3>Audio upload</h3>
                <p>Supported formats include WAV, MP3, M4A, MP4, and OGG. The system will transcribe and analyse automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload spoken response", type=["wav", "mp3", "m4a", "mp4", "ogg"])
        if uploaded_file is not None:
            is_valid_upload, validation_message = validate_audio_upload(uploaded_file)
            uploaded_file.seek(0)
            if is_valid_upload:
                st.audio(uploaded_file.read(), format=uploaded_file.type or "audio/wav")
                uploaded_file.seek(0)
                size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
                st.caption(f"Ready to analyse: `{uploaded_file.name}` ({size_mb:.2f} MB)")
            else:
                st.error(validation_message)
                uploaded_file = None

    analyse = st.button("Run premium AI analysis", type="primary", **full_width_kwargs(st.button))
    if analyse:
        if uploaded_file is None:
            st.warning("Upload an audio file first.")
            return
        if not concept_name:
            st.warning("Provide a concept name before starting the analysis.")
            return
        if not reference_answer:
            st.warning("Provide the reference answer before starting the analysis.")
            return

        with st.spinner("Transcribing, scoring, building charts, and generating your report..."):
            try:
                result = run_analysis(uploaded_file, concept_name, reference_answer, expected_concepts, db, user, config)
                st.session_state["latest_analysis"] = result
                st.session_state["auth_notice"] = "Analysis completed successfully. Review the full results now."
                navigate("results")
                st.rerun()
            except (AudioAnalysisError, SemanticAnalysisError, TranscriptionError, ValueError) as exc:
                st.error(f"Analysis failed: {exc}")
            except Exception as exc:  # pragma: no cover
                st.exception(exc)


def results_page(config: AppConfig) -> None:
    render_shell_header("Analysis results", "Review the latest scorecard, interpretation notes, and downloadable report.")
    show_notice()
    result = ensure_latest_analysis_theme_assets(st.session_state.get("latest_analysis"), config)
    if not result:
        st.markdown(
            """
            <div class="empty-state">
                <h3>No result ready yet</h3>
                <p>Run an audio analysis first to populate this page with transcription, charts, and report downloads.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to upload page", type="primary"):
            navigate("upload")
            st.rerun()
        return

    transcription_payload = result.get("transcription", {})
    if isinstance(transcription_payload, dict):
        transcription_text = transcription_payload.get("text", "")
    else:
        transcription_text = str(transcription_payload or "")

    scores = result["scoring"]["scores"]
    stat1, stat2, stat3, stat4, stat5 = st.columns(5)
    with stat1:
        render_stat_card("Understanding", scores["understanding"], "Semantic alignment to the reference", "violet")
    with stat2:
        render_stat_card("Fluency", scores["fluency"], "Pacing, pause, and filler balance", "cyan")
    with stat3:
        render_stat_card("Communication", scores["communication"], "Clarity, energy stability, and delivery", "green")
    with stat4:
        render_stat_card("Accuracy", scores["accuracy"], "Concept completeness and precision", "amber")
    with stat5:
        render_stat_card("Final score", scores["final"], "Weighted overall assessment", "violet")

    chart_col, meta_col = st.columns([1.15, 1], vertical_alignment="top")
    with chart_col:
        st.markdown("<div class='panel-title'>Visual analytics</div>", unsafe_allow_html=True)
        st.image(
            result["metric_chart_path"],
            caption="Scorecard distribution",
            **full_width_kwargs(st.image, image_fallback=True),
        )
        st.image(
            result["waveform_path"],
            caption="Waveform and energy envelope",
            **full_width_kwargs(st.image, image_fallback=True),
        )

    with meta_col:
        safe_concept_name = sanitize_html_text(result.get("concept_name", "Untitled concept"))
        safe_original_filename = sanitize_html_text(result["original_filename"])
        st.markdown(
            f"""
            <div class="glass-panel compact-panel">
                <h4>Run metadata</h4>
                <p><strong>Concept:</strong> {safe_concept_name}</p>
                <p><strong>File:</strong> {safe_original_filename}</p>
                <p><strong>Generated:</strong> {sanitize_html_text(result["evaluated_at"])}</p>
                <p><strong>Speech rate:</strong> {result["audio_metrics"]["speech_rate_wpm"]} WPM</p>
                <p><strong>Pause count:</strong> {result["audio_metrics"]["pause_count"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_history_download(result["report_path"], "Download PDF report")
        if st.button("Run another analysis", **full_width_kwargs(st.button)):
            navigate("upload")
            st.rerun()

    lower_left, lower_right = st.columns([1.1, 0.9], vertical_alignment="top")
    with lower_left:
        st.markdown("<div class='panel-title'>Transcription</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='text-panel'>{sanitize_html_text(transcription_text)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='panel-title'>Concept analysis</div>", unsafe_allow_html=True)
        matched = format_items_for_display(result["semantic"]["matched_concepts"]) or "No direct concept matches found."
        missing = format_items_for_display(result["semantic"]["missing_concepts"]) or "No major concept gaps detected."
        st.markdown(
            f"""
            <div class="glass-panel compact-panel">
                <p><strong>Similarity:</strong> {result["semantic"]["similarity_score"]:.2f}</p>
                <p><strong>Concept coverage:</strong> {result["semantic"]["concept_coverage"]:.2f}</p>
                <p><strong>Matched concepts:</strong> {matched}</p>
                <p><strong>Missing concepts:</strong> {missing}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lower_right:
        st.markdown("<div class='panel-title'>Strengths</div>", unsafe_allow_html=True)
        for item in result["scoring"]["strengths"]:
            st.markdown(f"<div class='bullet-card'>{sanitize_html_text(item)}</div>", unsafe_allow_html=True)
        if not result["scoring"]["strengths"]:
            st.markdown("<div class='bullet-card'>No strengths detected yet.</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel-title'>Weaknesses</div>", unsafe_allow_html=True)
        for item in result["scoring"]["weaknesses"]:
            st.markdown(f"<div class='bullet-card'>{sanitize_html_text(item)}</div>", unsafe_allow_html=True)
        if not result["scoring"]["weaknesses"]:
            st.markdown("<div class='bullet-card'>No weaknesses detected.</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel-title'>Improvement suggestions</div>", unsafe_allow_html=True)
        for item in result["scoring"]["suggestions"]:
            st.markdown(f"<div class='bullet-card accent'>{sanitize_html_text(item)}</div>", unsafe_allow_html=True)


def history_page(db: DatabaseManager, user_id: int) -> None:
    render_shell_header("History archive", "Browse previous analyses, compare scores, and download report evidence.")
    records = db.fetch_user_history(user_id)
    if not records:
        st.markdown(
            """
            <div class="empty-state">
                <h3>History is empty</h3>
                <p>Your completed analyses will appear here with downloadable reports and summary metrics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to upload page", key="history_empty_to_upload", type="primary", **full_width_kwargs(st.button)):
            navigate("upload")
            st.rerun()
        return

    history_df = build_history_dataframe(records)
    render_theme_table(history_df[["Created At", "Concept", "Final"]])

    selected_index = st.selectbox(
        "Choose a past analysis",
        options=list(history_df.index),
        format_func=lambda index: f"{history_df.loc[index, 'Concept']} • {history_df.loc[index, 'Created At']}",
    )
    selected = history_df.loc[selected_index]
    safe_selected_concept = sanitize_html_text(selected["Concept"])
    safe_selected_created = sanitize_html_text(selected["Created At"])
    preview_left, preview_right = st.columns([1.15, 0.85], vertical_alignment="top")
    with preview_left:
        st.markdown(
            f"""
            <div class="glass-panel compact-panel">
                <h4>{safe_selected_concept}</h4>
                <p><strong>Created:</strong> {safe_selected_created}</p>
                <p><strong>Final:</strong> {selected["Final"]:.1f}</p>
                <p><strong>Understanding:</strong> {selected["Understanding"]:.1f}</p>
                <p><strong>Fluency:</strong> {selected["Fluency"]:.1f}</p>
                <p><strong>Communication:</strong> {selected["Communication"]:.1f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with preview_right:
        render_history_download(str(selected["Report"]), "Download selected report")


def profile_page(db: DatabaseManager, user_id: int) -> None:
    render_shell_header("Profile", "Review account details, workspace activity, and your current analysis footprint.")
    user = db.get_user_by_id(user_id)
    if not user:
        st.error("User profile not found.")
        return

    summary = db.fetch_user_summary(user_id)
    records = db.fetch_user_history(user_id)
    history_df = build_history_dataframe(records) if records else pd.DataFrame()

    left, right = st.columns([1, 1.1], vertical_alignment="top")
    with left:
        safe_profile_name = sanitize_html_text(user["full_name"])
        safe_profile_email = sanitize_html_text(user["email"])
        safe_created_at = sanitize_html_text(user["created_at"])
        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-kicker">Account identity</div>
                <h3>{safe_profile_name}</h3>
                <p>{safe_profile_email}</p>
                <p>Member since: {safe_created_at}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        card1, card2, card3 = st.columns(3)
        with card1:
            render_stat_card("Analyses", float(summary["analysis_count"]), "Total completed runs", "violet")
        with card2:
            render_stat_card("Average", summary["average_final_score"], "Mean final score", "cyan")
        with card3:
            render_stat_card(
                "Top score",
                float(history_df["Final"].max()) if not history_df.empty else 0.0,
                "Best recorded result",
                "green",
            )

    st.markdown("<div class='panel-title'>Workspace summary</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-panel compact-panel">
            <p>Your account stores secure sign-in credentials, uploaded audio paths, generated reports, and scoring history in SQLite.</p>
            <p>The current workspace is optimized for repeated oral assessment sessions with a separate results page and downloadable evidence files.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    config = get_app_config()
    ensure_runtime_directories(config)
    st.session_state.setdefault("theme", "dark")
    load_css(config, st.session_state.get("theme", "dark"))

    db = get_database()
    auth_service = get_auth_service()
    SessionManager.bootstrap_state()
    if SessionManager.expire_if_needed():
        st.rerun()

    current_page = st.session_state.get("page", "landing")
    is_authenticated = SessionManager.is_authenticated()

    if is_authenticated and current_page in PUBLIC_PAGES:
        current_page = "dashboard"
        navigate(current_page)
    if not is_authenticated and current_page not in PUBLIC_PAGES:
        current_page = "landing"
        navigate(current_page)

    if not is_authenticated:
        if current_page == "landing":
            landing_page()
        elif current_page == "login":
            login_page(auth_service)
        elif current_page == "signup":
            signup_page(auth_service)
        return

    user = st.session_state["user"]
    selected_page = render_app_sidebar(user)
    if selected_page != st.session_state.get("page"):
        navigate(selected_page)
        st.rerun()
    else:
        current_page = st.session_state.get("page", "dashboard")

    if current_page not in APP_PAGES:
        navigate("dashboard")
        st.rerun()

    show_notice()
    render_workspace_nav(current_page)
    user_id = int(user["id"])
    if current_page == "dashboard":
        dashboard_page(db, user_id)
    elif current_page == "upload":
        upload_page(db, user, config)
    elif current_page == "results":
        results_page(config)
    elif current_page == "history":
        history_page(db, user_id)
    elif current_page == "profile":
        profile_page(db, user_id)


if __name__ == "__main__":
    main()
