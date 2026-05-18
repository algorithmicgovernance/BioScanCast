from .scoring import brier_score, log_score, range_brier_score, categorical_accuracy

def evaluate_binary(predictions):
    results = []
    for p in predictions:
        prob = p["prob"]
        outcome = p["outcome"]
        results.append({
            "brier": brier_score(prob, outcome),
            "log": log_score(prob, outcome)
        })
    return results


def evaluate_range(predictions):
    results = []
    for p in predictions:
        probs = p["probs"]
        true_idx = p["true_bucket"]
        results.append({
            "range_brier": range_brier_score(probs, true_idx)
        })
    return results


def evaluate_open(predictions):
    results = []
    for p in predictions:
        results.append({
            "confidence": p.get("confidence", None)
        })
    return results