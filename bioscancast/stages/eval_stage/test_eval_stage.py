from bioscancast.stages.eval_stage.evaluator import evaluate_binary

def test_binary():
    preds = [
        {"prob": 0.7, "outcome": 1},
        {"prob": 0.2, "outcome": 0}
    ]
    results = evaluate_binary(preds)
    assert len(results) == 2