from modules.audio_analysis import analyse_audio_file


def test_audio_analysis_extracts_metrics(sample_audio_file):
    result = analyse_audio_file(str(sample_audio_file), "this is a clear spoken answer about neural networks")

    assert result["duration_seconds"] > 1.5
    assert result["speech_rate_wpm"] > 0
    assert result["word_count"] >= 5
    assert 0 <= result["fluency_score"] <= 100
    assert 0 <= result["clarity_score"] <= 100
