"""Calculate metrics and plot the classification results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from degradations import PARAMETER_GRID
from explore_degradations import match_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
N_BINS = 10
SSIM_TOLERANCES = (0.01, 0.02)


def calibration_bins(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    if len(confidence) == 0 or len(confidence) != len(correct):
        raise ValueError("Calibration needs equally sized, nonempty arrays.")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must be finite and between zero and one.")
    # Intervals are [0, 0.1), [0.1, 0.2), ..., [0.9, 1].
    bin_ids = np.minimum((confidence * N_BINS).astype(int), N_BINS - 1)
    rows = []
    for index in range(N_BINS):
        mask = bin_ids == index
        count = int(mask.sum())
        average = float(confidence[mask].mean()) if count else np.nan
        accuracy = float(correct[mask].mean()) if count else np.nan
        rows.append({
            "bin_lower": index / N_BINS,
            "bin_upper": (index + 1) / N_BINS,
            "count": count,
            "mean_confidence": average,
            "accuracy": accuracy,
            "ece_contribution": count / len(confidence) * abs(accuracy - average) if count else 0.0,
        })
    return pd.DataFrame(rows)


def validate_predictions(frame: pd.DataFrame, test: pd.DataFrame) -> None:
    if len(frame) != len(test) or frame["image_id"].duplicated().any():
        raise ValueError("A condition must contain exactly one prediction per test image.")
    if set(frame["image_id"]) != set(test["image_id"]):
        raise ValueError("Prediction image IDs do not match the test split.")
    joined = frame.merge(test[["image_id", "lesion_id", "dx"]], on="image_id", validate="one_to_one", suffixes=("", "_split"))
    if not ((joined["true_label"] == joined["dx"]) & (joined["lesion_id"] == joined["lesion_id_split"])).all():
        raise ValueError("Prediction labels or lesion IDs differ from the test split.")
    probabilities = frame[[f"prob_{label}" for label in CLASSES]].to_numpy()
    confidence = frame["confidence"].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid class probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Class probabilities must sum to one.")
    if not np.allclose(confidence, probabilities.max(axis=1), atol=1e-6, rtol=0):
        raise ValueError("Confidence does not match the largest class probability.")
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    if not (frame["predicted_label"].to_numpy() == predicted).all():
        raise ValueError("Predicted labels do not match the probabilities.")


def calculate_metrics(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    true = frame["true_label"]
    predicted = frame["predicted_label"]
    confidence = frame["confidence"].to_numpy()
    bins = calibration_bins(confidence, (true == predicted).to_numpy())
    metrics = {
        "n_images": len(frame),
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, labels=list(CLASSES), average="macro", zero_division=0),
        "mean_confidence": float(confidence.mean()),
        "ece": float(bins["ece_contribution"].sum()),
    }
    return metrics, bins


def compare_matched_pairs(degraded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "image_id", "lesion_id", "true_label", "degradation", "parameter",
        "ssim", "predicted_label", "confidence",
    ]
    scores = degraded[columns].rename(columns={"true_label": "dx"}).copy()
    scores["split"] = "test"
    scores["correct"] = scores["predicted_label"] == scores["dx"]
    pairs = pd.concat(
        [match_pairs(scores, tolerance) for tolerance in SSIM_TOLERANCES],
        ignore_index=True,
    )
    total = scores["image_id"].nunique()
    summary = []
    kinds = sorted(PARAMETER_GRID)
    for tolerance in SSIM_TOLERANCES:
        for index, first in enumerate(kinds):
            for second in kinds[index + 1:]:
                group = pairs.loc[
                    (pairs["tolerance"] == tolerance)
                    & (pairs["degradation_a"] == first)
                    & (pairs["degradation_b"] == second)
                ]
                count = len(group)
                row = {
                    "tolerance": tolerance,
                    "degradation_a": first,
                    "degradation_b": second,
                    "n_images": count,
                    "n_lesions": group["lesion_id"].nunique(),
                    "n_unmatched_images": total - count,
                    "coverage": count / total,
                    "mean_ssim_a": group["ssim_a"].mean(),
                    "mean_ssim_b": group["ssim_b"].mean(),
                    "mean_ssim_gap": group["ssim_gap"].mean(),
                    "max_ssim_gap": group["ssim_gap"].max(),
                }
                for side in ("a", "b"):
                    correct = group[f"correct_{side}"]
                    confidence = group[f"confidence_{side}"]
                    row[f"accuracy_{side}"] = correct.mean() if count else np.nan
                    row[f"mean_confidence_{side}"] = confidence.mean()
                    row[f"ece_{side}"] = (
                        calibration_bins(confidence.to_numpy(), correct.to_numpy())["ece_contribution"].sum()
                        if count else np.nan
                    )
                row["accuracy_difference_a_minus_b"] = row["accuracy_a"] - row["accuracy_b"]
                row["confidence_difference_a_minus_b"] = row["mean_confidence_a"] - row["mean_confidence_b"]
                row["n_different_predictions"] = int((group["predicted_label_a"] != group["predicted_label_b"]).sum())
                row["n_both_correct"] = int((group["correct_a"] & group["correct_b"]).sum())
                row["n_only_a_correct"] = int((group["correct_a"] & ~group["correct_b"]).sum())
                row["n_only_b_correct"] = int((~group["correct_a"] & group["correct_b"]).sum())
                row["n_both_wrong"] = int((~group["correct_a"] & ~group["correct_b"]).sum())
                summary.append(row)
    return pairs, pd.DataFrame(summary)


def plot_performance(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    baseline = metrics.loc[metrics["degradation"] == "original"].iloc[0]
    labels = {
        "blur": "Blur radius (pixels)", "jpeg": "JPEG quality",
        "resolution": "Intermediate side length (pixels)",
        "brightness": "Brightness factor",
    }
    for ax, kind in zip(axes.flat, PARAMETER_GRID):
        group = metrics.loc[metrics["degradation"] == kind].sort_values("parameter")
        for column, label in (
            ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
            ("macro_f1", "Macro-F1"), ("mean_confidence", "Mean confidence"),
        ):
            ax.plot(group["parameter"], group[column], marker="o", label=label)
        ax.axhline(baseline["accuracy"], color="black", linestyle="--", linewidth=1, label="Original accuracy")
        ax.set(title=kind.capitalize(), xlabel=labels[kind], ylim=(0, 1.02))
        ax.set_xticks(group["parameter"])
        if kind == "blur":
            ax.tick_params(axis="x", labelrotation=45)
        if kind in ("jpeg", "resolution"):
            ax.invert_xaxis()
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.25)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3)
    fig.suptitle("Classification results by degradation level")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def class_metrics(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for label in CLASSES:
        group = frame.loc[frame["true_label"] == label]
        correct = (group["true_label"] == group["predicted_label"]).to_numpy()
        true_positives = int(correct.sum())
        predicted_count = int((frame["predicted_label"] == label).sum())
        recall = true_positives / len(group) if len(group) else np.nan
        precision = true_positives / predicted_count if predicted_count else 0.0
        rows.append({
            "true_label": label,
            "n_images": group["image_id"].nunique(),
            "n_lesions": group["lesion_id"].nunique(),
            "n_rows": len(group),
            "recall": recall,
            "precision": precision,
            "f1": 2 * true_positives / (len(group) + predicted_count) if len(group) + predicted_count else 0.0,
            "mean_confidence": group["confidence"].mean(),
            "ece": calibration_bins(group["confidence"].to_numpy(), correct)["ece_contribution"].sum() if len(group) else np.nan,
        })
    return rows


def plot_calibration(bins: pd.DataFrame, ece: float, output_path: Path) -> None:
    nonempty = bins.loc[bins["count"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].scatter(nonempty["mean_confidence"], nonempty["accuracy"], color="tab:blue", label="Nonempty bins")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title=f"Original images | ECE = {ece:.3f}")
    axes[0].legend(fontsize=9)
    centers = (bins["bin_lower"] + bins["bin_upper"]) / 2
    axes[1].bar(centers, bins["count"], width=0.09, color="tab:blue")
    axes[1].set(xlim=(0, 1), xlabel="Confidence", ylabel="Number of images", title="Images in each confidence interval")
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    original = pd.read_csv(results_dir / "original_predictions.csv")
    degraded = pd.read_csv(results_dir / "degraded_predictions.csv")
    splits = pd.read_csv(PROJECT_ROOT / "data" / "splits.csv")
    test = splits.loc[splits["split"] == "test"]
    expected = {(kind, float(value)) for kind, values in PARAMETER_GRID.items() for value in values}
    actual = set(degraded[["degradation", "parameter"]].itertuples(index=False, name=None))
    if actual != expected or not degraded["ssim"].between(-1, 1).all():
        raise ValueError("Unexpected degradation conditions or invalid SSIM scores.")
    conditions = [("original", np.nan, original)]
    conditions.extend((kind, parameter, group) for (kind, parameter), group in degraded.groupby(["degradation", "parameter"], sort=True))
    metric_rows = []
    class_rows = []
    bin_tables = []
    for kind, parameter, frame in conditions:
        validate_predictions(frame, test)
        metrics, bins = calculate_metrics(frame)
        metric_rows.append({"degradation": kind, "parameter": parameter, **metrics})
        bin_tables.append(bins.assign(degradation=kind, parameter=parameter))
        class_rows.extend({"degradation": kind, "parameter": parameter, **row} for row in class_metrics(frame))

    metrics = pd.DataFrame(metric_rows)
    bins = pd.concat(bin_tables, ignore_index=True)
    output_dir = results_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(class_rows).to_csv(output_dir / "class_metrics.csv", index=False, lineterminator="\n")
    bins.to_csv(output_dir / "calibration_bins.csv", index=False, lineterminator="\n")
    plot_performance(metrics, output_dir / "performance.png")
    plot_calibration(bins.loc[bins["degradation"] == "original"], metrics.iloc[0]["ece"], output_dir / "original_calibration.png")
    pairs, matched_summary = compare_matched_pairs(degraded)
    pairs.to_csv(output_dir / "matched_pairs.csv", index=False, lineterminator="\n")
    matched_summary.to_csv(output_dir / "matched_comparison.csv", index=False, lineterminator="\n")
    print(metrics.round(4).to_string(index=False))
    print("\nComparisons at similar SSIM:")
    print(matched_summary[[
        "tolerance", "degradation_a", "degradation_b", "n_images",
        "accuracy_a", "accuracy_b", "accuracy_difference_a_minus_b",
    ]].round(4).to_string(index=False))
    print("\nSaved metrics, matched comparisons, and figures in results/analysis/.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Analysis failed: {error}") from error
