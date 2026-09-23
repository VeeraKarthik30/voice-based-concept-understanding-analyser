# Voice-Based Concept Understanding Analyser
### Project Documentation

**Team Members**

| # | Name |
|---|------|
| 1 | D. Venkatesh |
| 2 | D. Karthik Reddy |
| 3 | E. Angel |
| 4 | G. Kousik |
| 5 | G. Pushpa Srivalli |

**Project Title:** Voice-Based Concept Understanding Analyser (VBCUA)
**Date:** July 2026

---

## Table of Contents

1. [Brainstorming & Ideation](#1-brainstorming--ideation)
2. [Requirement Analysis](#2-requirement-analysis)
3. [Project Design Phase](#3-project-design-phase)
4. [Project Planning Phase](#4-project-planning-phase)
5. [Project Development Phase](#5-project-development-phase)
6. [Project Testing](#6-project-testing)
7. [Project Documentation](#7-project-documentation)
8. [Project Demonstration](#8-project-demonstration)

---

## 1. Brainstorming & Ideation

### 1.1 Problem Identification

Traditional oral assessments in educational institutions suffer from three key limitations:

- **Subjectivity** — Different evaluators give different scores for the same answer.
- **Scalability** — Human grading cannot handle large cohorts simultaneously.
- **Feedback depth** — Verbal assessments rarely produce structured, actionable feedback.

Students who explain concepts verbally often lack clear metrics on *how well* they understood a topic versus *how well* they delivered it.

### 1.2 Brainstorming Session

The team conducted a collaborative brainstorming session using the following idea-generation approach:

| Idea | Feasibility | Impact | Priority |
|------|------------|--------|----------|
| Keyword-matching based grader | High | Low | Low |
| Sentiment analysis of speech | Medium | Low | Low |
| **Semantic similarity + audio analysis** | High | High | **Selected** |
| Full LLM-based grading (GPT-4) | Low (cost) | High | Deferred |
| ASR + grammar checker | Medium | Medium | Partial |

**Selected approach:** Combine OpenAI Whisper (speech-to-text) with Sentence-BERT (semantic similarity) and librosa (audio feature extraction) to produce a multi-dimensional concept understanding score.

### 1.3 Empathy Map

| Quadrant | Observations |
|----------|-------------|
| **Says** | "I know this concept but I can't express it clearly." |
| **Thinks** | "Am I being judged on what I know or how I speak?" |
| **Does** | Practices repeatedly without knowing which aspect to improve. |
| **Feels** | Anxious about oral tests; uncertain about performance. |

**Target users:** Students in higher education, corporate training, and professional certification programs.

### 1.4 Idea Prioritization

The Sentence-BERT + Whisper + librosa stack was selected because:
- Whisper is free, offline-capable, and multilingual.
- Sentence-BERT (`all-MiniLM-L6-v2`) is lightweight and accurate for short-form answers.
- librosa provides professional-grade audio feature extraction.
- All tools are open-source, enabling zero-cost deployment.

---

## 2. Requirement Analysis

### 2.1 Define Problem Statement

> **"How can we automatically evaluate a student's spoken explanation of a concept, measuring both conceptual understanding and verbal delivery quality, and provide structured, actionable feedback — at scale, objectively, and without human graders?"**

### 2.2 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-01 | User can register and log in with email/password authentication. |
| FR-02 | User can specify a concept name, reference answer, and expected concepts. |
| FR-03 | User can upload an audio file (WAV/MP3/M4A/OGG). |
| FR-04 | System transcribes audio to text using OpenAI Whisper. |
| FR-05 | System computes semantic similarity using Sentence-BERT. |
| FR-06 | System analyses audio for fluency and clarity metrics. |
| FR-07 | System generates a multi-dimensional scorecard (Understanding, Fluency, Communication, Accuracy, Final). |
| FR-08 | System generates a downloadable PDF report with all scores and charts. |
| FR-09 | User can view historical analyses with score trends. |
| FR-10 | User can toggle between light and dark UI themes. |

### 2.3 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Whisper model loads once and is cached (performance). |
| NFR-02 | All passwords are stored as bcrypt hashes (security). |
| NFR-03 | SQLite database stores all analyses with user isolation (data integrity). |
| NFR-04 | Application is containerisable via Docker (portability). |
| NFR-05 | UI must be accessible on desktop and tablet screen sizes (usability). |

### 2.4 Customer Journey Map

```
Student → Discovers VBCUA → Creates Account → Logs In
  → Selects Concept + Provides Reference Answer
  → Records / Uploads Voice Explanation
  → Waits for Analysis (~10-30 seconds)
  → Views Scorecard (Understanding, Fluency, Communication, Accuracy)
  → Reads Strengths, Weaknesses & Suggestions
  → Downloads PDF Report
  → Reviews Historical Trend on Dashboard
  → Improves & Re-submits
```

### 2.5 Data Flow Diagram

```
[User] --uploads--> [Audio File]
                         |
                         v
                  [Whisper ASR]  ---------------------> [Transcript Text]
                                                               |
              [Reference Answer] --> [Sentence-BERT] <--------+
                                          |                   |
                                          v                   v
                                   [Similarity Score]  [librosa Audio Analysis]
                                          |                   |
                                          +--------+----------+
                                                   v
                                         [Scoring Engine]
                                      (Understanding x 0.4
                                       + Fluency x 0.2
                                       + Communication x 0.2
                                       + Accuracy x 0.2)
                                                   |
                                +------------------+------------------+
                                v                  v                  v
                          [PDF Report]       [SQLite DB]        [Dashboard]
```

### 2.6 Solution Requirements

| Component | Technology Chosen | Reason |
|-----------|-------------------|--------|
| Speech-to-text | OpenAI Whisper `tiny` | Offline, fast, multilingual |
| Semantic similarity | sentence-transformers `all-MiniLM-L6-v2` | Lightweight, high accuracy |
| Audio feature extraction | librosa + soundfile | Industry-standard audio DSP |
| UI framework | Streamlit | Rapid ML application development |
| Database | SQLite | Zero-config, file-based |
| PDF generation | ReportLab | Programmatic, professional PDFs |
| Containerisation | Docker | Cloud-ready deployment |

### 2.7 Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10+ |
| UI | Streamlit | >= 1.37.0 |
| ASR | openai-whisper | >= 20240930 |
| NLP Embeddings | sentence-transformers | >= 3.0.1 |
| Audio Processing | librosa | >= 0.10.2 |
| Audio I/O | soundfile | >= 0.12.1 |
| Deep Learning | PyTorch | >= 2.2.0 |
| Data Manipulation | pandas | >= 2.2.2 |
| Numerical Computing | numpy | >= 1.26.4 |
| Visualisation | matplotlib | >= 3.8.4 |
| PDF Generation | reportlab | >= 4.2.2 |
| Testing | pytest | >= 8.2.0 |

---

## 3. Project Design Phase

### 3.1 Problem-Solution Fit

| Problem | Solution |
|---------|----------|
| Subjectivity in oral grading | Cosine similarity against a fixed reference answer removes human bias |
| No feedback on delivery | librosa audio analysis measures pauses, filler words, speech rate, and energy |
| Grader scalability | Fully automated pipeline; no human in the loop |
| No persistent records | SQLite + PDF export gives students downloadable evidence |
| Poor UX in similar tools | Streamlit multi-page app with dark/light theme, scorecard UI, and history tracking |

### 3.2 Proposed Solution

The **Voice-Based Concept Understanding Analyser** is a web application where:

1. A learner selects a **concept** and provides a **reference answer**.
2. They **upload** a voice recording of their explanation.
3. The system **transcribes** the recording with Whisper.
4. **Sentence-BERT** computes semantic similarity between transcription and reference answer.
5. **librosa** extracts audio features: duration, RMS energy, zero-crossing rate, pause count, speech rate, and filler words.
6. A **Scoring Engine** produces five scores (Understanding, Fluency, Communication, Accuracy, Final).
7. A **PDF report** with charts, scores, strengths, weaknesses, and suggestions is generated.
8. All analyses are stored in **SQLite** and visualised on a personal **Dashboard**.

### 3.3 Solution Architecture

```
Streamlit UI (app.py)
        |
        +---> Authentication Module (authentication/)
        |           |
        |           v
        |     SQLite DB (database/db.py)
        |
        +---> Upload Page
                    |
                    +---> modules/transcription.py  [Whisper ASR]
                    |
                    +---> modules/audio_analysis.py [librosa Features]
                    |
                    +---> modules/semantic_analysis.py [Sentence-BERT]
                    |
                    v
              modules/scoring_engine.py [Weighted Scorecard]
                    |
                    +---> utils/pdf_generator.py [ReportLab PDF]
                    |
                    +---> database/db.py [Persist Analysis]
                    |
                    v
              Dashboard / History / Results Pages
```

### 3.4 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `app.py` | Main Streamlit application — routing, page rendering, session management |
| `authentication/auth.py` | User registration, login, password hashing with bcrypt |
| `authentication/user_session.py` | Session state management (login/logout) |
| `database/db.py` | SQLite CRUD operations for users and analyses |
| `modules/transcription.py` | Whisper model loading (cached) and audio transcription |
| `modules/audio_analysis.py` | librosa-based audio feature extraction and fluency/clarity scoring |
| `modules/semantic_analysis.py` | Sentence-BERT embedding, cosine similarity, concept coverage |
| `modules/scoring_engine.py` | Weighted final score, strengths/weaknesses detection, suggestions |
| `modules/waveform.py` | Matplotlib waveform and metric chart generation |
| `utils/config.py` | Application path configuration (database, reports, uploads, assets) |
| `utils/helpers.py` | Utility functions: filler detection, slug generation, HTML sanitisation |
| `utils/pdf_generator.py` | ReportLab PDF report builder |

---

## 4. Project Planning Phase

### 4.1 Project Timeline

| Sprint | Duration | Tasks | Owner |
|--------|----------|-------|-------|
| Sprint 1 | Week 1 | Project setup, authentication module, SQLite schema | D. Venkatesh, D. Karthik Reddy |
| Sprint 2 | Week 2 | Whisper transcription module, audio analysis module | G. Kousik, G. Pushpa Srivalli |
| Sprint 3 | Week 3 | Semantic analysis module, scoring engine | E. Angel, D. Karthik Reddy |
| Sprint 4 | Week 4 | Streamlit UI (landing, login, signup, dashboard, upload, results) | All members |
| Sprint 5 | Week 5 | PDF generator, history page, waveform charts | D. Venkatesh, G. Pushpa Srivalli |
| Sprint 6 | Week 6 | Testing, Docker, documentation, demo preparation | All members |

### 4.2 Team Roles & Responsibilities

| Member | Role | Key Responsibilities |
|--------|------|----------------------|
| D. Venkatesh | Team Lead & Backend Dev | Architecture, database design, PDF generator, Docker |
| D. Karthik Reddy | ML Engineer | Scoring engine, semantic analysis, model integration |
| E. Angel | NLP & AI Engineer | Sentence-BERT pipeline, concept coverage logic, filler detection |
| G. Kousik | Audio Engineer | librosa audio analysis, waveform visualisation, Whisper integration |
| G. Pushpa Srivalli | Frontend & UI | Streamlit pages, CSS theming, dashboard, history page |

### 4.3 Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Whisper model too slow on CPU | Medium | High | Use `tiny` model; cache with `@lru_cache` |
| Large audio files causing timeouts | Medium | Medium | Validate upload size; limit to common formats |
| Sentence-BERT first-load latency | Low | Medium | Cache model with `@lru_cache(maxsize=1)` |
| SQLite concurrency issues | Low | Low | Streamlit single-user sessions |
| Inaccurate transcription for non-English | Medium | Medium | Document English-primary support |

---

## 5. Project Development Phase

### 5.1 Scoring Formula

The Scoring Engine computes five scores:

| Score | Formula |
|-------|---------|
| **Understanding** | `(Cosine Similarity x 0.7 + Concept Coverage x 0.3) x 100` |
| **Fluency** | `100 - (pause_count x 4.5) - (filler_rate x 100 x 20) - max(0, abs(speech_rate - 140) x 0.18)` |
| **Communication** | `(energy_stability x 55) + (ZCR_score x 25) + (filler_penalty x 20)` |
| **Accuracy** | `similarity_score - (missing_concepts x 4.0) - (filler_count x 1.5)` |
| **Final** | `Understanding x 0.4 + Fluency x 0.2 + Communication x 0.2 + Accuracy x 0.2` |

All scores are clamped to `[0.0, 100.0]`.

### 5.2 Code Layout & Structure

```
VoiceBasedConceptUnderstandingAnalyser/
|
+-- app.py                        # Main Streamlit entry point
+-- requirements.txt              # Python dependencies
+-- Dockerfile                    # Docker container definition
+-- packages.txt                  # System packages (ffmpeg etc.)
|
+-- authentication/
|   +-- auth.py                   # AuthService: signup, login, bcrypt
|   +-- user_session.py           # SessionManager: login/logout
|
+-- database/
|   +-- db.py                     # DatabaseManager: SQLite CRUD
|
+-- modules/
|   +-- transcription.py          # transcribe_audio() - Whisper ASR
|   +-- audio_analysis.py         # analyse_audio_file() - librosa features
|   +-- semantic_analysis.py      # analyse_concept_understanding() - SBERT
|   +-- scoring_engine.py         # ScoringEngine.generate_scorecard()
|   +-- waveform.py               # build_waveform_figure(), build_metric_chart()
|
+-- utils/
|   +-- config.py                 # AppConfig dataclass, directory management
|   +-- helpers.py                # detect_fillers(), safe_slug(), validation
|   +-- pdf_generator.py          # build_pdf_report() - ReportLab
|
+-- assets/
|   +-- styles_dark.css           # Dark theme CSS
|   +-- styles_light.css          # Light theme CSS
|
+-- reports/                      # Auto-created: PDFs, waveform charts
|   +-- uploads/                  # Auto-created: uploaded audio files
|
+-- tests/                        # pytest test suite
```

### 5.3 Key Functional Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | User Authentication | Secure signup/login with bcrypt-hashed passwords in SQLite |
| 2 | Audio Upload | Supports WAV, MP3, M4A, OGG; validated before processing |
| 3 | Whisper Transcription | Local ASR using `tiny` model; cached after first load |
| 4 | Semantic Similarity | Sentence-BERT cosine similarity vs reference answer |
| 5 | Concept Coverage | Tokenises expected concepts, checks presence in transcription |
| 6 | Audio Feature Extraction | Pause count, speech rate (WPM), filler words, RMS energy, ZCR |
| 7 | Multi-Dimensional Scoring | Five scores with configurable weights |
| 8 | Strengths & Weaknesses | Rule-based engine identifies specific performance areas |
| 9 | Improvement Suggestions | Auto-generated based on detected weaknesses |
| 10 | PDF Report Generation | ReportLab PDF with scores, waveform chart, metric chart |
| 11 | Score History & Trend | SQLite-persisted history with matplotlib trend line on Dashboard |
| 12 | Light / Dark Theme | CSS-driven theme toggle applied globally across all pages |

### 5.4 Code Readability & Reusability

- **Type annotations** used throughout all modules for clarity.
- **`@lru_cache`** applied to `_load_whisper_model()` and `_load_sentence_transformer()` to avoid reloading models.
- **`@st.cache_resource`** applied to `get_database()` and `get_auth_service()` for Streamlit-level caching.
- **Dataclasses** (`AppConfig`, `ScoreWeights`, `AnalysisPayload`) used for structured, typed configuration.
- **Modular design**: each `modules/` file has a single responsibility and can be tested independently.
- **`sanitize_html_text()`** applied to all user-generated content before HTML rendering.
- **`full_width_kwargs()`** utility ensures cross-Streamlit-version compatibility without duplication.

---

## 6. Project Testing

### 6.1 Test Strategy

The project uses **pytest** with unit tests for individual modules. Tests are in the `tests/` directory.

### 6.2 Test Cases

| Test ID | Module | Input | Expected Output | Pass/Fail |
|---------|--------|-------|-----------------|-----------|
| TC-01 | `semantic_analysis` | Learner answer identical to reference | Understanding score ~= 100 | Pass |
| TC-02 | `semantic_analysis` | Empty learner answer | `SemanticAnalysisError` raised | Pass |
| TC-03 | `semantic_analysis` | Empty reference answer | `SemanticAnalysisError` raised | Pass |
| TC-04 | `semantic_analysis` | All expected concepts present | concept_coverage = 100% | Pass |
| TC-05 | `audio_analysis` | WAV file with minimal pauses | fluency_score > 80 | Pass |
| TC-06 | `audio_analysis` | WAV file with filler words | filler_count > 0 | Pass |
| TC-07 | `scoring_engine` | All max sub-scores | final_score = 100.0 | Pass |
| TC-08 | `scoring_engine` | All zero sub-scores | final_score = 0.0 | Pass |
| TC-09 | `transcription` | Non-existent audio path | `TranscriptionError` raised | Pass |
| TC-10 | `helpers` | Text with "um", "uh", "like" | detect_fillers returns correct list | Pass |
| TC-11 | `config` | Default config call | All required directories resolved | Pass |
| TC-12 | `authentication` | Duplicate email signup | Returns failure message | Pass |

### 6.3 Performance Testing

| Metric | Target | Observed |
|--------|--------|----------|
| Whisper transcription (30s audio, tiny model) | < 60 seconds | ~15-25 seconds (CPU) |
| Sentence-BERT embedding (first load) | < 10 seconds | ~5-8 seconds |
| Sentence-BERT embedding (cached) | < 1 second | < 0.5 seconds |
| PDF report generation | < 5 seconds | ~1-2 seconds |
| Full pipeline (upload to scorecard) | < 90 seconds | ~30-60 seconds (CPU) |

### 6.4 Edge Cases Handled

- Empty audio signal -> `AudioAnalysisError` with descriptive message.
- Zero-duration audio -> `AudioAnalysisError` raised before analysis.
- Missing librosa -> fallback WAV loader using Python's `wave` module.
- Empty transcription result -> `TranscriptionError` raised.
- Zero-length concept list -> `concept_coverage` defaults to cosine similarity value.
- Division by zero in cosine similarity -> returns `0.0` safely.

---

## 7. Project Documentation

### 7.1 Setup & Installation

**Prerequisites:**
- Python 3.10 or higher
- `ffmpeg` installed on the system (required by Whisper)
- ~2 GB disk space for model weights

**Step-by-step:**

```bash
# 1. Clone the repository
git clone <repository-url>
cd VoiceBasedConceptUnderstandingAnalyser-main

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

**Docker:**

```bash
docker build -t vbcua .
docker run -p 8501:8501 vbcua
# Open http://localhost:8501
```

### 7.2 User Guide

| Step | Action |
|------|--------|
| 1 | Open the app at `http://localhost:8501` |
| 2 | Click **Get Started** to create an account |
| 3 | Log in with your credentials |
| 4 | Navigate to **Upload Audio** from the sidebar |
| 5 | Enter the concept name (e.g., "Photosynthesis") |
| 6 | Paste the reference/ideal answer |
| 7 | Optionally list expected concepts separated by commas |
| 8 | Upload your audio recording (WAV/MP3/M4A/OGG) |
| 9 | Click **Run Analysis** and wait for the pipeline |
| 10 | View your scorecard and download the PDF report |
| 11 | Check the **Dashboard** to track score trends over time |

### 7.3 Module API Reference

**`modules/transcription.py`**
```python
def transcribe_audio(audio_path: str) -> dict:
    # Returns: {"text": str, "language": str, "segments": list}
```

**`modules/semantic_analysis.py`**
```python
def analyse_concept_understanding(
    learner_answer: str,
    reference_answer: str,
    expected_concepts: str | None = None,
) -> dict:
    # Returns: {
    #   "similarity_score": float,   # 0-100
    #   "concept_coverage": float,   # 0-100
    #   "matched_concepts": list,
    #   "missing_concepts": list,
    #   "understanding_score": float # 0-100
    # }
```

**`modules/audio_analysis.py`**
```python
def analyse_audio_file(file_path: str, transcript_text: str) -> dict:
    # Returns: {
    #   "duration_seconds": float,
    #   "pause_count": int,
    #   "speech_rate_wpm": float,
    #   "filler_words": list,
    #   "filler_count": int,
    #   "fluency_score": float,  # 0-100
    #   "clarity_score": float,  # 0-100
    #   ...
    # }
```

**`modules/scoring_engine.py`**
```python
class ScoringEngine:
    def generate_scorecard(self, semantic_result, audio_result, transcription_text) -> dict:
        # Returns: {
        #   "scores": {"understanding", "fluency", "communication", "accuracy", "final"},
        #   "strengths": list[str],
        #   "weaknesses": list[str],
        #   "suggestions": list[str]
        # }
```

### 7.4 Configuration

All runtime paths are defined in `utils/config.py` via the `AppConfig` dataclass:

| Config Key | Default Path | Purpose |
|-----------|-------------|---------|
| `database_path` | `database/vbcua.sqlite3` | SQLite database file |
| `reports_dir` | `reports/` | PDF reports and chart images |
| `uploads_dir` | `reports/uploads/` | Uploaded audio files |
| `assets_dir` | `assets/` | CSS stylesheets |

---

## 8. Project Demonstration

### 8.1 Demo Planning

**Demo Objectives:**
1. Show end-to-end workflow from audio upload to scorecard.
2. Demonstrate semantic accuracy with a good vs. poor explanation of the same concept.
3. Show the dashboard's historical trend chart after multiple submissions.
4. Demonstrate PDF report download.

**Demo Scenario:**

> **Concept:** Newton's First Law of Motion
> **Reference Answer:** "An object at rest stays at rest, and an object in motion stays in motion at constant velocity unless acted upon by an external net force."
> **Expected Concepts:** inertia, velocity, external force, net force

- **Run 1 (Good answer):** Student gives a complete explanation covering all expected concepts -> high Understanding score.
- **Run 2 (Poor answer):** Student gives a vague answer -> low Understanding, flagged missing concepts.
- **Result:** Dashboard shows trend from low to high, demonstrating improvement tracking.

### 8.2 Proposed Features Demonstrated

| Feature | Demonstrated Via |
|---------|-----------------|
| Voice upload & transcription | Live audio upload, Whisper text output shown |
| Semantic scoring | Similarity score visible in scorecard |
| Concept coverage | Matched/missing concepts shown in results |
| Fluency analysis | Pause count, speech rate, filler words in PDF |
| Multi-dimensional scorecard | Metric chart in PDF |
| Waveform visualisation | Waveform image on Results page |
| PDF report | Downloaded and shown in demo |
| History & trend | Dashboard trend line after 2+ analyses |
| Light/Dark theme | Theme toggle demonstrated live |

### 8.3 Communication & Presentation Notes

- **Duration:** 10-15 minutes demo + 5 minutes Q&A
- **Presenter rotation:** Each team member presents one major section
- **Backup:** Pre-recorded analysis stored in session state in case of live Whisper delay

| Member | Presents |
|--------|----------|
| D. Venkatesh | Architecture, Database, Docker setup |
| D. Karthik Reddy | Scoring Engine & Semantic Analysis |
| E. Angel | NLP pipeline, Concept Coverage demo |
| G. Kousik | Audio Analysis, Waveform visualisation |
| G. Pushpa Srivalli | UI walkthrough, Dashboard, PDF download |

### 8.4 Project Executable

```bash
# Run locally
streamlit run app.py

# Run via Docker
docker build -t vbcua . && docker run -p 8501:8501 vbcua

# Run tests
pytest tests/ -v
```

**Live Access:** Open `http://localhost:8501` in any modern browser.

---

## Appendix A — Glossary

| Term | Definition |
|------|-----------|
| ASR | Automatic Speech Recognition — converting speech audio to text |
| Cosine Similarity | Angle-based similarity between two embedding vectors; 1.0 = identical |
| Sentence-BERT (SBERT) | Transformer model producing semantically meaningful sentence embeddings |
| Whisper | OpenAI's open-source multilingual ASR model |
| librosa | Python library for audio and music signal processing |
| WPM | Words Per Minute — a measure of speech rate |
| ZCR | Zero-Crossing Rate — rate at which a signal changes sign; related to pitch |
| RMS Energy | Root Mean Square energy — measure of audio loudness/volume |
| Filler Words | Non-meaningful words (um, uh, like, you know) that reduce clarity |
| SQLite | Lightweight, serverless, file-based relational database |

---

## Appendix B — References

1. OpenAI Whisper — https://github.com/openai/whisper
2. Sentence-Transformers — https://www.sbert.net/
3. librosa — https://librosa.org/
4. Streamlit — https://streamlit.io/
5. ReportLab — https://www.reportlab.com/
6. Project Template — https://github.com/Ravi-teja-777/AI-ML-and-GEN-AI-Track-Project-Template

---

*Document prepared by: D. Venkatesh, D. Karthik Reddy, E. Angel, G. Kousik, G. Pushpa Srivalli*
*Voice-Based Concept Understanding Analyser — July 2026*
