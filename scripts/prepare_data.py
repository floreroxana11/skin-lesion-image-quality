"""Prepare the HAM10000 train, validation, and test image lists."""

from pathlib import Path
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

EXPECTED_COUNTS = {"train": 6986, "validation": 1521, "test": 1508}
EXPECTED_TEST_CLASSES = {
    "akiec": 60, "bcc": 79, "bkl": 195, "df": 12,
    "mel": 175, "nv": 967, "vasc": 20,
}


def prepare_splits(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["image_id", "lesion_id", "dx"]
    if not set(columns).issubset(metadata.columns):
        raise ValueError("Metadata must contain image_id, lesion_id, and dx.")
    if metadata.empty or metadata[columns].isna().any().any():
        raise ValueError("Metadata is empty or required values are missing.")
    if metadata["image_id"].duplicated().any():
        raise ValueError("Image IDs must be unique.")
    if (metadata.groupby("lesion_id")["dx"].nunique() > 1).any():
        raise ValueError("Some lesions have conflicting diagnoses.")

    # Keep all images of the same lesion together.
    lesion_ids = sorted(metadata["lesion_id"].unique())
    random.Random(SEED).shuffle(lesion_ids)
    train_end = int(len(lesion_ids) * TRAIN_RATIO)
    validation_end = train_end + int(len(lesion_ids) * VALIDATION_RATIO)

    groups = {
        "train": set(lesion_ids[:train_end]),
        "validation": set(lesion_ids[train_end:validation_end]),
        "test": set(lesion_ids[validation_end:]),
    }
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            if groups[name] & groups[other]:
                raise ValueError(f"Lesions overlap between {name} and {other}.")

    split_by_lesion = {
        lesion_id: split
        for split, ids in groups.items()
        for lesion_id in ids
    }
    result = metadata[columns].copy()
    result["split"] = result["lesion_id"].map(split_by_lesion)
    if result["split"].isna().any():
        raise ValueError("Some images have not been assigned to a split.")
    if (result.groupby("lesion_id")["split"].nunique() != 1).any():
        raise ValueError("A lesion appears in more than one split.")
    if result["split"].value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Split sizes differ from the author's recorded results.")
    test_counts = result.loc[result["split"] == "test", "dx"].value_counts().to_dict()
    if test_counts != EXPECTED_TEST_CLASSES:
        raise ValueError("Test class counts differ from the published model card.")
    return result.sort_values("image_id").reset_index(drop=True)


def main() -> None:
    metadata_path = PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv"
    output_path = PROJECT_ROOT / "data" / "splits.csv"
    metadata = pd.read_csv(metadata_path)
    splits = prepare_splits(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_path, index=False, lineterminator="\n")

    print("Images per split:")
    print(splits["split"].value_counts().reindex(EXPECTED_COUNTS).to_string())
    print("\nImages per class and split:")
    print(pd.crosstab(splits["dx"], splits["split"])[list(EXPECTED_COUNTS)].to_string())
    print("\nNo lesions overlap between splits.")
    print("Split sizes and test class counts match the author's records.")
    print("Saved: data/splits.csv")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"Data preparation failed: {error}") from error
