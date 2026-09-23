import sqlite3

from database.db import DatabaseManager
from utils.helpers import AnalysisPayload


def test_database_enables_foreign_keys(tmp_path):
    db = DatabaseManager(tmp_path / "database.sqlite3")

    with db._connect() as conn:
        row = conn.execute("PRAGMA foreign_keys").fetchone()

    assert row[0] == 1


def test_database_saves_and_reads_analysis_history(tmp_path):
    db = DatabaseManager(tmp_path / "database.sqlite3")
    user_id = db.create_user("Ada Lovelace", "ada@example.com", "hashed-password")

    payload = AnalysisPayload(
        user_id=user_id,
        concept_name="Neural Networks",
        original_filename="answer.wav",
        stored_audio_path="/tmp/answer.wav",
        transcription="A neural network learns from data.",
        scores={"final": 88.4, "understanding": 90.0, "fluency": 82.0, "communication": 84.0, "accuracy": 86.0},
        report_path="/tmp/report.pdf",
    )
    db.save_analysis(payload)

    history = db.fetch_user_history(user_id)
    summary = db.fetch_user_summary(user_id)

    assert len(history) == 1
    assert history[0]["concept_name"] == "Neural Networks"
    assert summary["analysis_count"] == 1
    assert summary["average_final_score"] == 88.4


def test_database_rejects_orphan_analysis_when_foreign_keys_enabled(tmp_path):
    db = DatabaseManager(tmp_path / "database.sqlite3")

    with db._connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO analyses (
                    user_id, concept_name, original_filename, stored_audio_path,
                    transcription, scores_json, report_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (999, "Concept", "answer.wav", "/tmp/answer.wav", "text", "{}", "/tmp/report.pdf", "2026-07-01 10:00:00"),
            )
        except sqlite3.IntegrityError:
            failed = True
        else:
            failed = False

    assert failed is True
