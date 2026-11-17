"""Run the classifier on original or degraded images."""

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18

from degradations import PARAMETER_GRID, calculate_ssim, degrade, prepare_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
BATCH_SIZE = 16


def load_model() -> torch.nn.Module:
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    weights_path = PROJECT_ROOT / "models" / "weights" / "resnet.pth"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return model


def image_variants(original: Image.Image, degraded: bool):
    if not degraded:
        yield {}, original
        return
    for kind, parameters in PARAMETER_GRID.items():
        for parameter in parameters:
            variant = degrade(original, kind, parameter)
            yield {
                "degradation": kind,
                "parameter": parameter,
                "ssim": calculate_ssim(original, variant),
            }, variant


def predict_batch(model, tensors, records) -> list[dict]:
    probabilities = torch.softmax(model(torch.stack(tensors)), dim=1)
    results = []
    for record, scores in zip(records, probabilities.tolist()):
        predicted_index = max(range(len(CLASSES)), key=scores.__getitem__)
        predicted_class = CLASSES[predicted_index]
        result = {
            **record,
            "predicted_label": predicted_class,
            "confidence": scores[predicted_index],
            "correct": predicted_class == record["true_label"],
        }
        result.update({f"prob_{label}": score for label, score in zip(CLASSES, scores)})
        results.append(result)
    return results


def run_experiments(degraded: bool = False, split: str = "test") -> None:
    started = perf_counter()
    torch.set_num_threads(4)
    splits = pd.read_csv(PROJECT_ROOT / "data" / "splits.csv")
    if split not in ("test", "validation"):
        raise ValueError("Choose the test or validation split.")
    if (splits.groupby("lesion_id")["split"].nunique() > 1).any():
        raise ValueError("A lesion appears in more than one split.")
    selected_images = splits.loc[splits["split"] == split].sort_values("image_id")
    if selected_images.empty:
        raise ValueError(f"No {split} images found. Run prepare_data.py first.")
    if selected_images["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in the selected split.")
    if not set(selected_images["dx"]).issubset(CLASSES):
        raise ValueError("Unknown diagnosis in the selected image list.")
    model = load_model()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    image_paths = {}
    for path in (PROJECT_ROOT / "data" / "raw").rglob("*.jpg"):
        if path.stem in image_paths:
            raise ValueError(f"Duplicate image filename: {path.name}")
        image_paths[path.stem] = path

    rows = list(selected_images.itertuples(index=False))
    results = []
    tensors = []
    records = []
    variants_per_image = sum(map(len, PARAMETER_GRID.values())) if degraded else 1
    total = len(rows) * variants_per_image
    mode = "degraded" if degraded else "original"
    print(f"Running ResNet18 on {total} {mode} {split} images (CPU)...", flush=True)
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            if row.image_id not in image_paths:
                raise ValueError(f"Image file missing: {row.image_id}")
            with Image.open(image_paths[row.image_id]) as image:
                original = prepare_image(image)
            for variant_info, variant in image_variants(original, degraded):
                tensors.append(preprocess(variant))
                records.append({
                    "image_id": row.image_id,
                    "lesion_id": row.lesion_id,
                    "true_label": row.dx,
                    **variant_info,
                })
                if len(tensors) == BATCH_SIZE:
                    results.extend(predict_batch(model, tensors, records))
                    tensors.clear()
                    records.clear()
            if index % 25 == 0 or index == len(rows):
                print(f"Prepared {index}/{len(rows)} source images", flush=True)
        if tensors:
            results.extend(predict_batch(model, tensors, records))

    if len(results) != total:
        raise ValueError(f"Expected {total} predictions, obtained {len(results)}.")

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    filename = "degraded_predictions.csv" if degraded else "original_predictions.csv"
    if split == "validation":
        filename = f"validation_{filename}"
    output_path = output_dir / filename
    pd.DataFrame(results).to_csv(output_path, index=False, lineterminator="\n")
    elapsed = perf_counter() - started
    print(f"Saved: results/{filename}")
    print(f"Elapsed time: {elapsed:.1f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degraded", action="store_true", help="Evaluate all degradation levels and calculate SSIM.")
    parser.add_argument("--split", choices=("test", "validation"), default="test", help="Image split to process (default: test).")
    args = parser.parse_args()
    try:
        run_experiments(args.degraded, args.split)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"Experiment failed: {error}") from error


if __name__ == "__main__":
    main()
