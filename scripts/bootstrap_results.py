"""Estimate uncertainty in the matched and reliability comparisons."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_RESAMPLES = 1000
SEED = 42


def lesion_weights(lesion_ids: pd.Series, rng: np.random.Generator):
    codes, lesions = pd.factorize(lesion_ids, sort=True)
    if (codes < 0).any() or len(lesions) < 2:
        raise ValueError("Bootstrap requires at least two lesions with valid IDs.")
    for _ in range(N_RESAMPLES):
        sampled = rng.integers(0, len(lesions), size=len(lesions))
        counts = np.bincount(sampled, minlength=len(lesions))
        yield counts[codes]


def interval(values) -> tuple[float, float]:
    values = np.asarray(values)
    if len(values) < 0.9 * N_RESAMPLES or not np.isfinite(values).all():
        raise ValueError("Not enough valid bootstrap resamples.")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_matched(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.duplicated(["tolerance", "degradation_a", "degradation_b", "image_id"]).any():
        raise ValueError("Duplicate matched images in a comparison.")
    if not ((pairs["ssim_a"] - pairs["ssim_b"]).abs() <= pairs["tolerance"]).all():
        raise ValueError("A matched pair exceeds its SSIM tolerance.")
    rng = np.random.default_rng(SEED)
    output = []
    for (tolerance, first, second), group in pairs.groupby(
        ["tolerance", "degradation_a", "degradation_b"], sort=True,
    ):
        correct_a = (group["predicted_label_a"] == group["dx"]).to_numpy(dtype=float)
        correct_b = (group["predicted_label_b"] == group["dx"]).to_numpy(dtype=float)
        differences = {
            "accuracy": correct_a - correct_b,
            "mean_confidence": (group["confidence_a"] - group["confidence_b"]).to_numpy(),
        }
        resampled = {metric: [] for metric in differences}
        # All rows belonging to a sampled lesion receive the same weight.
        for weights in lesion_weights(group["lesion_id"], rng):
            for metric, values in differences.items():
                resampled[metric].append(float(np.average(values, weights=weights)))
        for metric, values in differences.items():
            lower, upper = interval(resampled[metric])
            output.append({
                "tolerance": tolerance,
                "degradation_a": first,
                "degradation_b": second,
                "metric": metric,
                "n_images": group["image_id"].nunique(),
                "n_lesions": group["lesion_id"].nunique(),
                "difference_a_minus_b": float(values.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "includes_zero": lower <= 0 <= upper,
                "n_resamples": N_RESAMPLES,
                "seed": SEED,
            })
        print(f"Matched comparison: {first} vs {second}, tolerance {tolerance}", flush=True)
    return pd.DataFrame(output)


def error_metrics(correct: np.ndarray, probability_correct: np.ndarray, weights: np.ndarray) -> dict:
    errors = 1 - correct
    probability_error = 1 - probability_correct
    return {
        "error_auroc": roc_auc_score(errors, probability_error, sample_weight=weights),
        "error_average_precision": average_precision_score(errors, probability_error, sample_weight=weights),
        "brier_score": float(np.average((probability_correct - correct) ** 2, weights=weights)),
    }


def bootstrap_reliability(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.duplicated(["image_id", "degradation", "parameter"]).any():
        raise ValueError("Duplicate degraded variants in reliability predictions.")
    correct = (predictions["true_label"] == predictions["predicted_label"]).to_numpy(dtype=int)
    probabilities = {name: predictions[f"probability_correct_{name}"].to_numpy() for name in "ABCD"}
    if not all(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in probabilities.values()):
        raise ValueError("Invalid estimated probabilities.")
    comparisons = (("A", "B"), ("C", "D"))
    baseline = {name: error_metrics(correct, p, np.ones(len(correct))) for name, p in probabilities.items()}
    resampled = {(first, second, metric): [] for first, second in comparisons for metric in baseline[first]}
    rng = np.random.default_rng(SEED)
    for index, weights in enumerate(lesion_weights(predictions["lesion_id"], rng), start=1):
        weighted_correct = np.dot(correct, weights)
        if weighted_correct > 0 and weighted_correct < weights.sum():
            metrics = {name: error_metrics(correct, p, weights) for name, p in probabilities.items()}
            for first, second in comparisons:
                for metric in metrics[first]:
                    resampled[first, second, metric].append(metrics[second][metric] - metrics[first][metric])
        if index % 100 == 0:
            print(f"Reliability bootstrap: {index}/{N_RESAMPLES}", flush=True)

    output = []
    for (first, second, metric), values in resampled.items():
        lower, upper = interval(values)
        output.append({
            "baseline": first,
            "with_type": second,
            "metric": metric,
            "n_images": predictions["image_id"].nunique(),
            "n_lesions": predictions["lesion_id"].nunique(),
            "n_variants": len(predictions),
            "difference_with_type_minus_baseline": baseline[second][metric] - baseline[first][metric],
            "ci_lower": lower,
            "ci_upper": upper,
            "includes_zero": lower <= 0 <= upper,
            "improvement_direction": "negative" if metric == "brier_score" else "positive",
            "n_valid_resamples": len(values),
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
        })
    return pd.DataFrame(output)


def main() -> None:
    results = PROJECT_ROOT / "results"
    pairs = pd.read_csv(results / "analysis" / "matched_pairs.csv")
    predictions = pd.read_csv(results / "reliability" / "predictions.csv")
    splits = pd.read_csv(PROJECT_ROOT / "data" / "splits.csv")
    if splits["image_id"].duplicated().any() or (splits.groupby("lesion_id")["split"].nunique() > 1).any():
        raise ValueError("Invalid image or lesion separation in splits.csv.")
    test = splits.loc[splits["split"] == "test"].set_index("image_id")
    for frame, label in ((pairs, "dx"), (predictions, "true_label")):
        if frame.empty or not frame["image_id"].isin(test.index).all():
            raise ValueError("Bootstrap input contains images outside the test split.")
        expected = test.loc[frame["image_id"]]
        if not np.array_equal(frame["lesion_id"].to_numpy(), expected["lesion_id"].to_numpy()):
            raise ValueError("Bootstrap input contains mismatched lesion IDs.")
        if not np.array_equal(frame[label].to_numpy(), expected["dx"].to_numpy()):
            raise ValueError("Bootstrap input contains mismatched labels.")
    matched = bootstrap_matched(pairs)
    reliability = bootstrap_reliability(predictions)
    output_dir = results / "bootstrap"
    output_dir.mkdir(exist_ok=True)
    matched.to_csv(output_dir / "matched_intervals.csv", index=False, lineterminator="\n")
    reliability.to_csv(output_dir / "reliability_intervals.csv", index=False, lineterminator="\n")
    print("\nMatched accuracy differences:")
    print(matched.loc[matched["metric"] == "accuracy", [
        "tolerance", "degradation_a", "degradation_b", "difference_a_minus_b",
        "ci_lower", "ci_upper", "includes_zero",
    ]].round(5).to_string(index=False))
    print("\nReliability differences after adding degradation type:")
    print(reliability[[
        "baseline", "with_type", "metric", "difference_with_type_minus_baseline",
        "ci_lower", "ci_upper", "includes_zero",
    ]].round(5).to_string(index=False))
    print("\nSaved 95% percentile intervals in results/bootstrap/.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Bootstrap analysis failed: {error}") from error
