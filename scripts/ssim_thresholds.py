"""Choose SSIM thresholds on validation and evaluate them on test images."""

from pathlib import Path
from time import perf_counter
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from analyze_results import validate_predictions
from bootstrap_results import N_RESAMPLES, SEED, interval, lesion_weights
from estimate_reliability import build_estimator, load_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELTAS = (0.02, 0.05)
GROUPS = ("blur", "brightness", "jpeg", "resolution",
          "brightness_dark", "brightness_bright", "pooled")
BIN_EDGES = np.round(np.arange(0.70, 1.001, 0.02), 2)
MIN_BIN_SIZE = 50


def select_group(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "pooled":
        return frame
    if group.startswith("brightness_"):
        brightness = frame.loc[frame["degradation"] == "brightness"]
        mask = brightness["parameter"] < 1 if group.endswith("dark") else brightness["parameter"] > 1
        return brightness.loc[mask]
    return frame.loc[frame["degradation"] == group]


def choose_thresholds(validation: pd.DataFrame, baseline: float) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        frame = select_group(validation, group)
        estimator = build_estimator(["ssim"], False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(frame, frame["correct"])
        scale = estimator.named_steps["features"].named_transformers_["numeric"]
        logistic = estimator.named_steps["logistic"]
        slope = float(logistic.coef_[0, 0] / scale.scale_[0])
        intercept = float(logistic.intercept_[0] - slope * scale.mean_[0])
        lower, upper = float(frame["ssim"].min()), float(frame["ssim"].max())
        endpoints = estimator.predict_proba(pd.DataFrame({"ssim": [lower, upper]}))[:, 1]
        for delta in DELTAS:
            target = baseline - delta
            threshold = np.nan
            status = "not reached"
            if endpoints[0] >= target:
                threshold, status = lower, "reached"
            elif slope > 0 and endpoints[1] >= target:
                threshold = float((np.log(target / (1 - target)) - intercept) / slope)
                threshold = float(np.clip(threshold, lower, upper))
                status = "reached"
            rows.append({
                "group": group, "delta": delta, "threshold": threshold, "status": status,
                "validation_original_accuracy": baseline, "target_accuracy": target,
                "observed_ssim_min": lower, "observed_ssim_max": upper,
                "logistic_intercept": intercept, "logistic_ssim_slope": slope,
                "increasing_fit": slope > 0,
                "predicted_accuracy_at_min": endpoints[0],
                "predicted_accuracy_at_max": endpoints[1],
                "validation_variants": len(frame),
                "validation_images": frame["image_id"].nunique(),
                "validation_lesions": frame["lesion_id"].nunique(),
            })
        print(f"Validation threshold fit: {group}", flush=True)
    return pd.DataFrame(rows)


def iso_ssim_curves(test: pd.DataFrame, baseline: float, output: Path) -> pd.DataFrame:
    rows = []
    for group in GROUPS[:-1]:
        frame = select_group(test, group)
        for index, (lower, upper) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
            below_upper = frame["ssim"] <= upper if index == len(BIN_EDGES) - 2 else frame["ssim"] < upper
            cell = frame.loc[(frame["ssim"] >= lower) & below_upper]
            if len(cell) >= MIN_BIN_SIZE:
                rows.append({
                    "group": group, "ssim_lower": lower, "ssim_upper": upper,
                    "ssim_midpoint": (lower + upper) / 2, "n_variants": len(cell),
                    "n_images": cell["image_id"].nunique(), "n_lesions": cell["lesion_id"].nunique(),
                    "accuracy": cell["correct"].mean(), "mean_confidence": cell["confidence"].mean(),
                })
    curves = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for row, groups in enumerate((GROUPS[:4], ("brightness_dark", "brightness_bright"))):
        for group in groups:
            cells = curves.loc[curves["group"] == group]
            for column, metric in enumerate(("accuracy", "mean_confidence")):
                axes[row, column].plot(cells["ssim_midpoint"], cells[metric], marker="o", label=group)
        for column, title in enumerate(("Accuracy", "Mean confidence")):
            ax = axes[row, column]
            ax.axhline(baseline, linestyle="--", color="black", label="Original accuracy")
            ax.set(xlabel="SSIM bin midpoint", ylabel=title, ylim=(0, 1.02))
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Test results in fixed SSIM bins (at least 50 variants per bin)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "iso_ssim_curves.png", dpi=150)
    plt.close(fig)
    return curves


def evaluate_thresholds(test: pd.DataFrame, originals: pd.DataFrame,
                        thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = originals.sort_values("image_id").reset_index(drop=True).copy()
    originals["correct"] = (originals["true_label"] == originals["predicted_label"]).astype(int)
    original_correct = originals["correct"].to_numpy(dtype=float)
    baseline = original_correct.mean()
    lookup = originals.set_index("image_id")
    test = test.copy()
    test["original_correct"] = lookup.loc[test["image_id"], "correct"].to_numpy()
    image_codes = pd.Index(originals["image_id"]).get_indexer(test["image_id"])
    if (image_codes < 0).any():
        raise ValueError("A degraded image has no original prediction.")
    threshold_index = thresholds.set_index(["group", "delta"])
    configs = []
    rows = []
    for delta in DELTAS:
        for group in GROUPS:
            frame = select_group(test, group)
            for policy in (("global",) if group == "pooled" else ("type_specific", "global")):
                source = "pooled" if policy == "global" else group
                chosen = threshold_index.loc[(source, delta)]
                threshold = chosen["threshold"]
                accepted = frame.loc[frame["ssim"] >= threshold] if np.isfinite(threshold) else frame.iloc[:0]
                indices = accepted.index.to_numpy()
                count = len(accepted)
                accuracy = accepted["correct"].mean()
                paired_original = accepted["original_correct"].mean()
                row = {
                    "group": group, "policy": policy, "delta": delta,
                    "threshold": threshold, "threshold_status": chosen["status"],
                    "n_available_variants": len(frame), "n_accepted_variants": count,
                    "n_accepted_images": accepted["image_id"].nunique(),
                    "n_accepted_lesions": accepted["lesion_id"].nunique(),
                    "coverage": count / len(frame), "accepted_accuracy": accuracy,
                    "original_test_accuracy": baseline,
                    "accuracy_drop_vs_all_originals": baseline - accuracy,
                    "paired_original_accuracy": paired_original,
                    "accuracy_drop_vs_paired_originals": paired_original - accuracy,
                    "drop_exceeds_delta": bool(baseline - accuracy > delta) if count else np.nan,
                    "paired_drop_exceeds_delta": bool(paired_original - accuracy > delta) if count else np.nan,
                }
                configs.append(indices)
                rows.append(row)
    samples = [{"all": [], "paired": []} for _ in rows]
    correct = test["correct"].to_numpy(dtype=float)
    paired = test["original_correct"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    for index, weights in enumerate(lesion_weights(originals["lesion_id"], rng), start=1):
        resampled_baseline = np.average(original_correct, weights=weights)
        variant_weights = weights[image_codes]
        for indices, sample in zip(configs, samples):
            selected_weights = variant_weights[indices]
            if selected_weights.sum():
                accuracy = np.average(correct[indices], weights=selected_weights)
                sample["all"].append(float(resampled_baseline - accuracy))
                sample["paired"].append(float(np.average(paired[indices] - correct[indices], weights=selected_weights)))
        if index % 250 == 0:
            print(f"Threshold bootstrap: {index}/{N_RESAMPLES}", flush=True)
    for row, sample in zip(rows, samples):
        for key, label in (("all", "all_originals"), ("paired", "paired_originals")):
            valid = len(sample[key])
            lower, upper = interval(sample[key]) if valid >= 0.9 * N_RESAMPLES else (np.nan, np.nan)
            row[f"drop_vs_{label}_ci_lower"] = lower
            row[f"drop_vs_{label}_ci_upper"] = upper
        row["n_valid_resamples"] = len(sample["all"])
        row["n_resamples"] = N_RESAMPLES
        row["seed"] = SEED
    evaluation = pd.DataFrame(rows)
    comparisons = []
    for delta in DELTAS:
        primary = evaluation.loc[(evaluation["delta"] == delta) & evaluation["group"].isin(GROUPS[:4])]
        for policy, group in primary.groupby("policy", sort=True):
            failures = group.loc[group["drop_exceeds_delta"] == True]
            comparisons.append({
                "delta": delta, "policy": policy,
                "n_accepted_variants": group["n_accepted_variants"].sum(),
                "n_groups_exceeding_delta": len(failures),
                "n_accepted_variants_in_failing_groups": failures["n_accepted_variants"].sum(),
                "n_groups_exceeding_paired_delta": int((group["paired_drop_exceeds_delta"] == True).sum()),
                "n_accepted_variants_in_paired_failing_groups": group.loc[
                    group["paired_drop_exceeds_delta"] == True, "n_accepted_variants"
                ].sum(),
                "n_groups_without_acceptance": int((group["n_accepted_variants"] == 0).sum()),
            })
    return evaluation, pd.DataFrame(comparisons)


def main() -> None:
    started = perf_counter()
    results = PROJECT_ROOT / "results"
    output = results / "thresholds"
    splits = pd.read_csv(PROJECT_ROOT / "data/splits.csv")
    if splits["image_id"].duplicated().any() or (splits.groupby("lesion_id")["split"].nunique() > 1).any():
        raise ValueError("Invalid image or lesion separation in splits.csv.")
    validation_split = splits.loc[splits["split"] == "validation"]
    validation = load_predictions(results / "validation_degraded_predictions.csv", validation_split)
    validation_originals = pd.read_csv(results / "validation_original_predictions.csv")
    validate_predictions(validation_originals, validation_split)
    baseline = (validation_originals["true_label"] == validation_originals["predicted_label"]).mean()
    thresholds = choose_thresholds(validation, baseline)
    output.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(output / "validation_thresholds.csv", index=False, lineterminator="\n")
    print("Thresholds fixed from validation. Loading test predictions...", flush=True)
    test_split = splits.loc[splits["split"] == "test"]
    test = load_predictions(results / "degraded_predictions.csv", test_split)
    originals = pd.read_csv(results / "original_predictions.csv")
    validate_predictions(originals, test_split)
    test_baseline = (originals["true_label"] == originals["predicted_label"]).mean()
    curves = iso_ssim_curves(test, test_baseline, output)
    evaluation, comparison = evaluate_thresholds(test, originals, thresholds)
    for name, frame in (("iso_ssim_bins.csv", curves), ("test_evaluation.csv", evaluation),
                        ("policy_comparison.csv", comparison)):
        frame.to_csv(output / name, index=False, lineterminator="\n")
    print(f"Validation original accuracy: {baseline:.4f}; test original accuracy: {test_baseline:.4f}")
    print(thresholds[["group", "delta", "threshold", "status", "increasing_fit"]].round(5).to_string(index=False))
    print(evaluation.loc[(evaluation["delta"] == DELTAS[0]) & evaluation["group"].isin(GROUPS[:4]),
                         ["group", "policy", "coverage", "accepted_accuracy",
                          "accuracy_drop_vs_all_originals", "accuracy_drop_vs_paired_originals"]].round(4).to_string(index=False))
    print("Saved: results/thresholds/ (four CSV tables and one figure)")
    print(f"Elapsed time: {perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ConvergenceWarning) as error:
        raise SystemExit(f"Threshold analysis failed: {error}") from error
