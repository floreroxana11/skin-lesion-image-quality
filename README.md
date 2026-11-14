# Impact of Image Quality on Prediction Reliability in AI-Based Skin Lesion Classification

This project studies how image quality affects the performance and reliability
of an AI model for skin lesion classification.

## Objective

The main objective is to compare versions of the same skin lesion image that
have similar quality scores but different types of degradation. We compare the
predictions and confidence scores for these images.

## Experiments

- Apply blur, JPEG compression, reduced resolution, and lighting changes to
  skin lesion images.
- Compare the model's predictions on original and degraded images.
- Compare different degradation types at similar image quality scores.

We evaluate accuracy, confidence, and calibration.

## Dataset

The project uses HAM10000, which contains 10,015 dermoscopic images from seven
skin lesion categories. The metadata includes image identifiers, lesion
identifiers, and diagnosis labels. Images of the same lesion must remain in the
same data partition when separating training, validation, and test data.

Reference: Tschandl, P., Rosendahl, C., & Kittler, H. (2018).
[The HAM10000 dataset, a large collection of multi-source dermatoscopic images
of common pigmented skin lesions](https://doi.org/10.1038/sdata.2018.161).

## Installation

The project uses Python 3.12. Create a virtual environment and install the required libraries:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
and place the following files and folders under `data/raw/`:

```text
data/raw/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

The raw data is excluded from Git.

## Usage

Check the metadata and image inventory:

```powershell
.\.venv\Scripts\python.exe scripts/check_dataset.py
```

This script checks that the images listed in the metadata are present and shows
the number of images in each class.
