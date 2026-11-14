"""Check HAM10000 images and metadata."""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def check_dataset(data_dir: Path) -> None:
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")
    if metadata.empty:
        raise ValueError("The metadata table is empty.")
    if metadata[sorted(required)].isna().any().any():
        raise ValueError("Missing image ID, lesion ID, or diagnosis.")
    if metadata["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in metadata.")
    if set(metadata["dx"]) != EXPECTED_CLASSES:
        raise ValueError("The metadata does not contain the seven expected classes.")
    if (metadata.groupby("lesion_id")["dx"].nunique() > 1).any():
        raise ValueError("Some lesions have conflicting diagnosis labels.")

    image_paths = [
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    image_ids = [path.stem for path in image_paths]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image filenames in the dataset folders.")

    expected_ids = set(metadata["image_id"])
    actual_ids = set(image_ids)
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing or extra:
        raise ValueError(
            f"Image inventory mismatch: {len(missing)} missing, {len(extra)} extra. "
            f"Missing examples: {sorted(missing)[:5]}; "
            f"extra examples: {sorted(extra)[:5]}"
        )

    print(f"Metadata rows: {len(metadata)}")
    print(f"Image files: {len(image_paths)}")
    print(f"Unique lesions: {metadata['lesion_id'].nunique()}")
    print("\nImages per diagnosis:")
    print(metadata["dx"].value_counts().sort_index().to_string())
    print("\nDataset inventory checks passed.")
    print("Image contents have not been decoded by this check.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw",
        help="Folder containing HAM10000_metadata.csv and image subfolders.",
    )
    args = parser.parse_args()
    try:
        check_dataset(args.data_dir)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        parser.exit(1, f"Dataset check failed: {error}\n")


if __name__ == "__main__":
    main()
