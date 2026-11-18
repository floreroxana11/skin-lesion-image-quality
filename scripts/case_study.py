"""Illustrate the experiments on 35 validation images."""

from pathlib import Path
from time import perf_counter
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
import torch
from torchvision import transforms

from analyze_results import compare_matched_pairs
from degradations import degrade, prepare_image
from estimate_reliability import VARIANTS, build_estimator, load_predictions
from explore_degradations import save_examples
from run_experiments import BATCH_SIZE, load_model, predict_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 0.01


def predict_originals(selected: pd.DataFrame, paths: dict) -> pd.DataFrame:
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = load_model()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    results, tensors, records = [], [], []
    with torch.inference_mode():
        for row in selected.itertuples(index=False):
            with Image.open(paths[row.image_id]) as image:
                tensors.append(preprocess(prepare_image(image)))
            records.append({
                "image_id": row.image_id, "lesion_id": row.lesion_id,
                "true_label": row.dx,
            })
            if len(tensors) == BATCH_SIZE:
                results.extend(predict_batch(model, tensors, records))
                tensors.clear()
                records.clear()
        if tensors:
            results.extend(predict_batch(model, tensors, records))
    frame = pd.DataFrame(results)
    frame["degradation"] = "original"
    frame["parameter"] = float("nan")
    frame["ssim"] = 1.0
    return frame


def estimate_correctness(training: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    examples = examples.copy()
    for name, settings in VARIANTS.items():
        estimator = build_estimator(**settings)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(training, training["correct"].astype(int))
        index = list(estimator.named_steps["logistic"].classes_).index(1)
        examples[f"probability_correct_{name}"] = estimator.predict_proba(examples)[:, index]
    return examples


def save_walkthrough(examples: pd.DataFrame, originals: pd.DataFrame, paths: dict,
                     output: Path) -> None:
    if len(examples) != 2:
        raise ValueError("Expected at least two matched pairs with different correctness.")
    fig, axes = plt.subplots(2, 3, figsize=(12, 11))
    original_rows = originals.set_index("image_id")
    for axes_row, row in zip(axes, examples.itertuples(index=False)):
        with Image.open(paths[row.image_id]) as image:
            original = prepare_image(image)
        base = original_rows.loc[row.image_id]
        axes_row[0].imshow(original)
        axes_row[0].set_title(
            f"{row.image_id} | original\nTrue: {row.dx} | predicted: {base.predicted_label}"
            f"\nConfidence: {base.confidence:.3f}", fontsize=10,
        )
        for column, side in enumerate(("a", "b"), start=1):
            kind = getattr(row, f"degradation_{side}")
            parameter = getattr(row, f"parameter_{side}")
            axes_row[column].imshow(degrade(original, kind, parameter))
            axes_row[column].set_title(
                f"{kind}: {parameter:g} | SSIM: {getattr(row, f'ssim_{side}'):.3f}"
                f"\nTrue: {row.dx} | predicted: {getattr(row, f'predicted_label_{side}')}"
                f"\nConfidence: {getattr(row, f'confidence_{side}'):.3f}"
                f" | P(correct), D: {getattr(row, f'probability_correct_D_{side}'):.3f}",
                fontsize=10,
            )
        for ax in axes_row:
            ax.axis("off")
    fig.suptitle("Illustrative validation examples at similar SSIM")
    fig.subplots_adjust(top=0.87, bottom=0.03, hspace=0.45, wspace=0.08)
    fig.savefig(output / "walkthrough.png", dpi=150)
    plt.close(fig)


def main() -> None:
    started = perf_counter()
    selected = pd.read_csv(PROJECT_ROOT / "data/validation_selection.csv")
    splits = pd.read_csv(PROJECT_ROOT / "data/splits.csv")
    if (splits.groupby("lesion_id")["split"].nunique() > 1).any():
        raise ValueError("A lesion appears in more than one split.")
    validation = splits.loc[splits["split"] == "validation"]
    if len(selected) != 35 or selected["image_id"].nunique() != 35:
        raise ValueError("Expected 35 distinct selected images.")
    verified = selected.merge(validation, on=["image_id", "lesion_id", "dx", "split"],
                              validate="one_to_one")
    if len(verified) != 35 or selected["lesion_id"].nunique() != 35:
        raise ValueError("Selected images must belong to 35 distinct validation lesions.")
    development = load_predictions(
        PROJECT_ROOT / "results/validation_degraded_predictions.csv", validation,
    )
    selected = selected.sort_values("image_id")
    examples = development.loc[development["image_id"].isin(selected["image_id"])].copy()
    # Exclude all photographs of the example lesions when fitting the estimators.
    training = development.loc[~development["lesion_id"].isin(selected["lesion_id"])].copy()
    if set(training["lesion_id"]) & set(examples["lesion_id"]):
        raise ValueError("Example and estimator-training lesions overlap.")
    paths = {}
    wanted = set(selected["image_id"])
    for path in (PROJECT_ROOT / "data/raw").rglob("*.jpg"):
        if path.stem in wanted:
            if path.stem in paths:
                raise ValueError(f"Duplicate image filename: {path.name}")
            paths[path.stem] = path
    if set(paths) != wanted:
        raise ValueError("Some selected image files are missing.")
    output = PROJECT_ROOT / "results/case_study"
    print("Predicting the 35 original validation images...", flush=True)
    originals = predict_originals(selected, paths)
    examples = estimate_correctness(training, examples)
    pairs, summary = compare_matched_pairs(examples)
    pairs = pairs.loc[pairs["tolerance"] == TOLERANCE].copy()
    pairs["split"] = "validation"
    for side in ("a", "b"):
        keys = ["image_id", "degradation", "parameter"]
        probabilities = examples[keys + [f"probability_correct_{name}" for name in VARIANTS]]
        probabilities = probabilities.rename(columns={
            column: f"{column}_{side}" for column in probabilities if column != "image_id"
        })
        pairs = pairs.merge(probabilities,
                            on=["image_id", f"degradation_{side}", f"parameter_{side}"],
                            validate="many_to_one")
    summary = summary.loc[summary["tolerance"] == TOLERANCE].copy()
    candidates = pairs.loc[pairs["correct_a"] != pairs["correct_b"]]
    walkthrough = candidates.sort_values(
        ["image_id", "pair", "parameter_a", "parameter_b"]
    ).head(2)
    if len(examples) != 700 or len(summary) != 6:
        raise ValueError("Expected 700 variants and six type comparisons.")
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("predictions.csv", pd.concat([originals, examples], ignore_index=True)),
        ("matched_pairs.csv", pairs), ("matched_comparison.csv", summary),
        ("walkthrough_pairs.csv", walkthrough),
        ("estimator_training.csv", training[["image_id", "lesion_id", "true_label"]].drop_duplicates()),
    ):
        frame.to_csv(output / name, index=False, lineterminator="\n")
    save_walkthrough(walkthrough, originals, paths, output)
    first = selected.iloc[0]
    with Image.open(paths[first.image_id]) as image:
        save_examples(prepare_image(image), first.image_id, output / "ssim_grid.png")
    print(summary[["degradation_a", "degradation_b", "n_images",
                   "accuracy_a", "accuracy_b"]].round(3).to_string(index=False))
    print(f"Estimator fitting: {training['image_id'].nunique()} validation images, "
          f"{training['lesion_id'].nunique()} lesions.")
    print("Saved: results/case_study/ (five CSV tables and two figures)")
    print(f"Elapsed time: {perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, ConvergenceWarning) as error:
        raise SystemExit(f"Case study failed: {error}") from error
