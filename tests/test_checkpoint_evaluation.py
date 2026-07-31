from src.evaluation import checkpoint_evaluation


def test_checkpoint_evaluation_reports_pool_aggregates_and_wilson_bounds(tmp_path, monkeypatch):
    validation = tmp_path / "validation.json"
    holdout = tmp_path / "holdout.json"
    validation.write_text(
        '{"opponents": [{"label": "validation-a", "deck_path": "a.csv", "model_path": "a.zip"}]}',
        encoding="utf-8",
    )
    holdout.write_text(
        '{"opponents": [{"label": "holdout-a", "deck_path": "b.csv", "model_path": "b.zip"}]}',
        encoding="utf-8",
    )

    def fake_evaluate(*args, **kwargs):
        return (30, 20, 0, 0, 0, 0, 0, 0, 0), {"mean_turns": 12.0}

    monkeypatch.setattr(checkpoint_evaluation, "evaluate_vs_opponent", fake_evaluate)
    output = tmp_path / "report.json"
    report = checkpoint_evaluation.evaluate_checkpoint(
        candidate_model="candidate.zip",
        candidate_deck="candidate.csv",
        validation_manifest=validation,
        holdout_manifest=holdout,
        games_per_opponent=50,
        output_path=output,
        global_steps=250_000,
        training_pool=[{"label": "train-a", "model_path": "train.zip", "weight": 0.1}],
    )

    assert output.is_file()
    assert report["validation"]["macro_win_rate"] == 0.6
    assert report["holdout"]["worst_win_rate"] == 0.6
    assert 0.0 < report["validation"]["worst_wilson_lower_bound_95"] < 0.6
    assert report["training_pool"][0]["label"] == "train-a"
