"""Explore degradation levels on validation images."""

from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from degradations import PARAMETER_GRID, calculate_ssim, degrade, prepare_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
IMAGES_PER_CLASS = 5


def select_images(splits: pd.DataFrame) -> pd.DataFrame:
    validation = splits.loc[splits["split"] == "validation"].sort_values("image_id")
    if validation.empty or validation["dx"].nunique() != 7:
        raise ValueError("Expected seven classes in the validation split.")
    if (splits.groupby("lesion_id")["split"].nunique() > 1).any():
        raise ValueError("A lesion appears in more than one split.")
    selected = []
    for _, group in validation.groupby("dx", sort=True):
        lesions = group.drop_duplicates("lesion_id")
        if len(lesions) < IMAGES_PER_CLASS:
            raise ValueError("Not enough distinct validation lesions in a class.")
        selected.append(lesions.sample(n=IMAGES_PER_CLASS, random_state=SEED))
    return pd.concat(selected).sort_values(["dx", "image_id"]).reset_index(drop=True)


def save_examples(original: Image.Image, image_id: str, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 6, figsize=(14, 10))
    for row, (kind, parameters) in enumerate(PARAMETER_GRID.items()):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title("Original | SSIM 1.000", fontsize=9)
        axes[row, 0].axis("off")
        for column, parameter in enumerate(parameters, start=1):
            variant = degrade(original, kind, parameter)
            score = calculate_ssim(original, variant)
            axes[row, column].imshow(variant)
            axes[row, column].set_title(f"{kind}: {parameter}\nSSIM {score:.3f}", fontsize=9)
            axes[row, column].axis("off")
    fig.suptitle(f"Validation image {image_id}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def match_pairs(scores: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    matches = []
    keys = ["image_id", "lesion_id", "dx", "split"]
    for first, second in combinations(sorted(scores["degradation"].unique()), 2):
        pairs = scores.loc[scores["degradation"] == first].merge(
            scores.loc[scores["degradation"] == second],
            on=keys, suffixes=("_a", "_b"),
        )
        pairs["ssim_gap"] = (pairs["ssim_a"] - pairs["ssim_b"]).abs()
        pairs = pairs.loc[pairs["ssim_gap"] <= tolerance]
        # Keep the closest pair for each image and pair of degradation types.
        pairs = pairs.sort_values(
            ["ssim_gap", "image_id", "parameter_a", "parameter_b"]
        ).drop_duplicates("image_id")
        pairs["tolerance"] = tolerance
        pairs["pair"] = f"{first} vs {second}"
        matches.append(pairs)
    return pd.concat(matches, ignore_index=True)


def main() -> None:
    splits = pd.read_csv(PROJECT_ROOT / "data" / "splits.csv")
    selected = select_images(splits)
    image_paths = {}
    for path in (PROJECT_ROOT / "data" / "raw").rglob("*.jpg"):
        if path.stem in image_paths:
            raise ValueError(f"Duplicate image filename: {path.name}")
        image_paths[path.stem] = path

    results = []
    example = None
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        if row.image_id not in image_paths:
            raise ValueError(f"Image file missing: {row.image_id}")
        with Image.open(image_paths[row.image_id]) as image:
            original = prepare_image(image)
        if example is None:
            example = (original, row.image_id)
        for kind, parameters in PARAMETER_GRID.items():
            for parameter in parameters:
                variant = degrade(original, kind, parameter)
                results.append({
                    "image_id": row.image_id,
                    "lesion_id": row.lesion_id,
                    "dx": row.dx,
                    "split": row.split,
                    "degradation": kind,
                    "parameter": parameter,
                    "ssim": calculate_ssim(original, variant),
                })
        print(f"Processed {index}/{len(selected)} validation images", flush=True)

    output_dir = PROJECT_ROOT / "results" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(PROJECT_ROOT / "data" / "validation_selection.csv", index=False, lineterminator="\n")
    scores = pd.DataFrame(results)
    scores.to_csv(output_dir / "ssim_scores.csv", index=False, lineterminator="\n")
    matches = pd.concat(
        [match_pairs(scores, tolerance) for tolerance in (0.01, 0.02)],
        ignore_index=True,
    )
    matches.to_csv(output_dir / "matched_pairs.csv", index=False, lineterminator="\n")
    save_examples(*example, output_dir / "degradation_examples.png")
    print("\nSSIM by degradation and parameter:")
    print(scores.groupby(["degradation", "parameter"])["ssim"].agg(["min", "mean", "max"]).round(3).to_string())
    print("\nMatched images out of 35, by SSIM tolerance:")
    coverage = matches.groupby(["pair", "tolerance"])["image_id"].nunique().unstack(fill_value=0)
    print(coverage.to_string())
    print("\nSaved: data/validation_selection.csv")
    print("Saved: results/validation/ssim_scores.csv")
    print("Saved: results/validation/matched_pairs.csv")
    print("Saved: results/validation/degradation_examples.png")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Degradation exploration failed: {error}") from error
