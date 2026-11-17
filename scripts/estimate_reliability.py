"""Compare ways of estimating prediction reliability."""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_results import validate_predictions
from degradations import PARAMETER_GRID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "A": {"numeric": ["ssim"], "use_type": False},
    "B": {"numeric": ["ssim"], "use_type": True},
    "C": {"numeric": ["ssim", "confidence"], "use_type": False},
    "D": {"numeric": ["ssim", "confidence"], "use_type": True},
    "E": {"numeric": ["confidence"], "use_type": False},
}


def load_predictions(path: Path, partition: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {(kind, float(value)) for kind, values in PARAMETER_GRID.items() for value in values}
    actual = set(frame[["degradation", "parameter"]].itertuples(index=False, name=None))
    if actual != expected or not frame["ssim"].between(-1, 1).all():
        raise ValueError(f"Unexpected degradation conditions or invalid SSIM: {path.name}")
    for _, condition in frame.groupby(["degradation", "parameter"]):
        validate_predictions(condition, partition)
    frame["correct"] = (frame["true_label"] == frame["predicted_label"]).astype(int)
    if frame["correct"].nunique() != 2:
        raise ValueError("Both correct and incorrect predictions are needed.")
    return frame.sort_values(["image_id", "degradation", "parameter"]).reset_index(drop=True)


def build_estimator(numeric: list[str], use_type: bool) -> Pipeline:
    transformers = [("numeric", StandardScaler(), numeric)]
    if use_type:
        transformers.append((
            "type", OneHotEncoder(drop="first", sparse_output=False), ["degradation"],
        ))
    return Pipeline([
        ("features", ColumnTransformer(transformers, remainder="drop")),
        ("logistic", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, tol=1e-6, random_state=42,
        )),
    ])


def evaluate_reliability(correct: np.ndarray, probability_correct: np.ndarray) -> dict:
    if not np.isfinite(probability_correct).all() or ((probability_correct < 0) | (probability_correct > 1)).any():
        raise ValueError("Estimated probabilities must be finite and between zero and one.")
    errors = 1 - correct
    probability_error = 1 - probability_correct
    return {
        "error_auroc": roc_auc_score(errors, probability_error) if np.unique(errors).size == 2 else np.nan,
        "error_average_precision": average_precision_score(errors, probability_error) if errors.sum() else np.nan,
        "brier_score": brier_score_loss(correct, probability_correct, pos_label=1),
    }


def analyze_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in predictions.groupby("true_label", sort=True):
        target = group["correct"].to_numpy()
        for name in VARIANTS:
            rows.append({
                "true_label": label,
                "variant": name,
                "n_images": group["image_id"].nunique(),
                "n_lesions": group["lesion_id"].nunique(),
                "n_variants": len(group),
                "error_rate": float(1 - target.mean()),
                **evaluate_reliability(target, group[f"probability_correct_{name}"].to_numpy()),
            })
    return pd.DataFrame(rows)


def analyze_deferral(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.sort_values(["image_id", "degradation", "parameter"]).reset_index(drop=True).copy()
    frame["tie_order"] = np.random.default_rng(42).permutation(len(frame))
    rows = []
    class_rows = []
    for name in VARIANTS:
        ordered = frame.sort_values([f"probability_correct_{name}", "tie_order"], ascending=[False, True])
        for rejection in (0.0, 0.1, 0.2):
            n_rejected = int(round(len(ordered) * rejection))
            n_retained = len(ordered) - n_rejected
            retained = ordered.iloc[:n_retained]
            rejected = ordered.iloc[n_retained:]
            n_errors = int((1 - ordered["correct"]).sum())
            n_errors_removed = int((1 - rejected["correct"]).sum())
            rows.append({
                "variant": name,
                "requested_rejection": rejection,
                "n_total": len(ordered),
                "n_retained": n_retained,
                "n_rejected": n_rejected,
                "coverage": n_retained / len(ordered),
                "retained_accuracy": retained["correct"].mean(),
                "n_errors_total": n_errors,
                "n_errors_removed": n_errors_removed,
                "fraction_errors_removed": n_errors_removed / n_errors if n_errors else np.nan,
            })
            for label, group in ordered.groupby("true_label", sort=True):
                kept = retained.loc[retained["true_label"] == label]
                removed = rejected.loc[rejected["true_label"] == label]
                class_rows.append({
                    "variant": name,
                    "requested_rejection": rejection,
                    "true_label": label,
                    "n_images": group["image_id"].nunique(),
                    "n_lesions": group["lesion_id"].nunique(),
                    "n_total": len(group),
                    "n_retained": len(kept),
                    "n_rejected": len(removed),
                    "coverage": len(kept) / len(group),
                    "retained_accuracy": kept["correct"].mean(),
                    "n_errors_removed": int((1 - removed["correct"]).sum()),
                })
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


def plot_deferral(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name in VARIANTS:
        group = summary.loc[summary["variant"] == name].sort_values("requested_rejection")
        axes[0].plot(group["requested_rejection"], group["retained_accuracy"], marker="o", label=name)
        axes[1].plot(group["requested_rejection"], group["fraction_errors_removed"], marker="o", label=name)
    axes[0].set(title="Accuracy among retained predictions", ylabel="Accuracy")
    axes[1].set(title="Errors among deferred predictions", ylabel="Fraction of all errors deferred")
    for ax in axes:
        ax.set(xlabel="Fraction of predictions deferred", ylim=(0, 1), xticks=[0, 0.1, 0.2])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.25)
        ax.legend(title="Variant")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    splits = pd.read_csv(PROJECT_ROOT / "data" / "splits.csv")
    if splits["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in the split list.")
    if (splits.groupby("lesion_id")["split"].nunique() > 1).any():
        raise ValueError("A lesion appears in more than one split.")
    validation = splits.loc[splits["split"] == "validation"]
    test = splits.loc[splits["split"] == "test"]
    results_dir = PROJECT_ROOT / "results"
    development = load_predictions(results_dir / "validation_degraded_predictions.csv", validation)
    evaluation = load_predictions(results_dir / "degraded_predictions.csv", test)
    if set(development["lesion_id"]) & set(evaluation["lesion_id"]):
        raise ValueError("Estimator training and test lesions overlap.")

    train_target = development["correct"].to_numpy()
    test_target = evaluation["correct"].to_numpy()
    predictions = evaluation[[
        "image_id", "lesion_id", "true_label", "predicted_label",
        "degradation", "parameter", "ssim", "confidence", "correct",
    ]].copy()
    metric_rows = []
    for name, settings in VARIANTS.items():
        estimator = build_estimator(**settings)
        # Fit preprocessing and the estimator only on validation data.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(development, train_target)
        classes = estimator.named_steps["logistic"].classes_
        correct_index = list(classes).index(1)
        probability_correct = estimator.predict_proba(evaluation)[:, correct_index]
        predictions[f"probability_correct_{name}"] = probability_correct
        features = settings["numeric"] + (["degradation"] if settings["use_type"] else [])
        metric_rows.append({
            "variant": name,
            "features": " + ".join(features),
            "training_images": development["image_id"].nunique(),
            "training_lesions": development["lesion_id"].nunique(),
            "training_variants": len(development),
            "test_images": evaluation["image_id"].nunique(),
            "test_lesions": evaluation["lesion_id"].nunique(),
            "test_variants": len(evaluation),
            "test_error_rate": float((1 - test_target).mean()),
            **evaluate_reliability(test_target, probability_correct),
        })
        print(f"Variant {name} trained on validation and evaluated on test.", flush=True)

    metrics = pd.DataFrame(metric_rows)
    output_dir = results_dir / "reliability"
    output_dir.mkdir(exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False, lineterminator="\n")
    predictions.to_csv(output_dir / "predictions.csv", index=False, lineterminator="\n")
    analyze_subgroups(predictions).to_csv(output_dir / "class_metrics.csv", index=False, lineterminator="\n")
    deferral, class_deferral = analyze_deferral(predictions)
    deferral.to_csv(output_dir / "deferral.csv", index=False, lineterminator="\n")
    class_deferral.to_csv(output_dir / "class_deferral.csv", index=False, lineterminator="\n")
    plot_deferral(deferral, output_dir / "deferral.png")
    print("\nHigher AUROC and Average Precision are better; lower Brier score is better.")
    print(metrics[["variant", "features", "error_auroc", "error_average_precision", "brier_score"]].round(4).to_string(index=False))
    print(f"\nTest error rate (reference for Average Precision): {metrics.iloc[0]['test_error_rate']:.4f}")
    print("Saved: results/reliability/metrics.csv")
    print("Saved: results/reliability/predictions.csv")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ConvergenceWarning) as error:
        raise SystemExit(f"Reliability estimation failed: {error}") from error
