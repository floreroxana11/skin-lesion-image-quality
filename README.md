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

The project uses a ResNet18 model trained on HAM10000. Download the weights from
[Hugging Face](https://huggingface.co/miloszkrawczyk/skin-lesion-classifier/resolve/645de358497899dbed2e4ca3ed137305404131f7/resnet.pth?download=true)
and save them as `models/weights/resnet.pth`. This file is excluded from Git.

Model source:
[miloszkrawczyk/skin-lesion-classifier](https://huggingface.co/miloszkrawczyk/skin-lesion-classifier).
The model uses 224 x 224 RGB images and ImageNet normalization.
Its license is CC BY-NC 4.0 (non-commercial use).

## Usage

Check the metadata and image inventory:

```powershell
.\.venv\Scripts\python.exe scripts/check_dataset.py
```

This script checks that the images listed in the metadata are present and shows
the number of images in each class.

Prepare the image lists for training, validation, and testing:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
```

This saves the image lists in `data/splits.csv`, using the
[author's split](https://github.com/miloszkrawczyk/melanoma-classifier/blob/b3ed99dc4b370ea97f799f096e0c148bf5d82bdd/src/data.py)
with seed 42: 6,986 training, 1,521 validation, and 1,508 test images.
Images of the same lesion stay in the same split.

Check the degradation levels on validation images:

```powershell
.\.venv\Scripts\python.exe scripts/explore_degradations.py
```

This selects five images from different lesions in each class, using seed 42.
It applies the degradations and calculates SSIM, without running the classifier.
The selected images are listed in `data/validation_selection.csv`.
The scores, examples, and pairs with SSIM differences of up to 0.01 or 0.02
are saved in `results/validation/`.

Run on all original test images:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py
```

This saves the labels, predictions, confidence, and class probabilities in
`results/original_predictions.csv`.

Run the 20 degraded variants of each test image:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --degraded
```

This saves 30,160 predictions in `results/degraded_predictions.csv`, together
with the degradation type, parameter, and SSIM score.

Run the same degradations on validation images:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --degraded --split validation
```

This saves 30,420 predictions in `results/validation_degraded_predictions.csv`.
They are used to train the reliability estimators.

Analyze the saved predictions:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_results.py
```

This calculates accuracy, balanced accuracy, Macro-F1, mean confidence, and ECE
for the originals and each degradation level. ECE compares confidence with
accuracy in ten confidence groups. Tables and graphs are saved in
`results/analysis/`, including metrics for each class.

The script also compares different degradations of the same image at similar
SSIM. For each pair of types, it selects the closest pair of variants within
0.01 or 0.02 SSIM, using only the scores. The pairs are saved in
`matched_pairs.csv`. Their accuracy, confidence, ECE, and matching counts are
saved in `matched_comparison.csv`. Differences are the first type minus the
second. Each type comparison can contain a different subset of images.

Estimate prediction reliability:

```powershell
.\.venv\Scripts\python.exe scripts/estimate_reliability.py
```

This trains logistic regression to estimate whether a ResNet prediction is correct:

| Variant | Inputs |
| --- | --- |
| A | SSIM |
| B | SSIM + degradation type |
| C | SSIM + ResNet confidence |
| D | SSIM + degradation type + ResNet confidence |
| E | ResNet confidence only |

All five variants are trained on validation data and evaluated on the same test
data. Numeric inputs are standardized using validation data. B is compared with
A, and D with C, to check whether the degradation type helps. E provides a
comparison using confidence alone.

The output is an estimated probability of correctness. Higher error AUROC and
Average Precision are better; lower Brier score is better. Metrics, results by
class, and estimated probabilities are saved in `results/reliability/`.
AUROC is left empty for classes without both correct and incorrect predictions;
Average Precision is left empty when there are no errors.

The script also sets aside the least reliable 10% or 20% of predictions and
measures accuracy on those remaining. It saves the results in `deferral.csv`,
results by class in `class_deferral.csv`, and a graph in `deferral.png`.
Ties use a random order with seed 42. The rule is applied to all predictions
together, so the percentage set aside can differ between classes.

Variant E, deferral, and class-level analyses were added after examining the
initial test results and are reported as exploratory analyses.

Calculate confidence intervals for the comparisons:

```powershell
.\.venv\Scripts\python.exe scripts/bootstrap_results.py
```

This uses 1,000 bootstrap samples with seed 42. Whole lesions are sampled with
replacement, keeping their images and variants together. Both sides of each
comparison use the same sampled lesions. ResNet and the fitted estimators are
not retrained.

The 95% intervals are saved in `results/bootstrap/`. They cover accuracy and
confidence differences between matched degradation types, and AUROC, Average
Precision, and Brier differences for B minus A and D minus C. Positive
differences are better for AUROC and Average Precision; negative differences
are better for Brier. The intervals are exploratory and are not adjusted for
multiple comparisons.

These analysis scripts use saved predictions without running ResNet again.
The formulas, model settings, and references are in
[Experimental Methodology](docs/methodology.md).

## Small case study

```powershell
.\.venv\Scripts\python.exe scripts/case_study.py
```

This uses the 35 selected validation images and their 700 degraded variants.
ResNet runs only on the 35 originals; variant predictions are read from the
saved validation results. Estimators A-E are fitted on the remaining validation
lesions, excluding all photographs of the example lesions.

Tables and figures are saved in `results/case_study/`. `predictions.csv` lists
the originals and variants; estimated correctness is provided for the variants.
`conditions.csv` gives the Experiment 1 metrics for the originals, each of the
20 conditions, and each degradation type.
`matched_comparison.csv` summarizes the six type comparisons at tolerance 0.01.
`reliability.csv` and `deferral.csv` give the Experiment 3 results for
estimators A-E on the 700 variants.
`estimator_training.csv` records the images used to fit the example estimators.

The walkthrough shows the first two matched pairs sorted by image ID where
only one prediction is correct. Ties are ordered by type pair, then ascending
parameters. `walkthrough_pairs.csv` records those pairs. The SSIM grid uses the
first selected image sorted by image ID. These examples are illustrative,
not representative of the full dataset. This case study is an exploratory
addition made after the initial test results.

## SSIM thresholds

Generate predictions for the original validation images, then run the analysis:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --split validation
.\.venv\Scripts\python.exe scripts/ssim_thresholds.py
```

Thresholds are chosen on validation data for an allowed accuracy drop of 2
percentage points, with 5 points as a second comparison. The script fits
correctness against SSIM for each degradation type and for all types together.
It reports `not reached` when the fitted accuracy does not reach the target
within the observed SSIM range.

The fixed thresholds are evaluated on saved test predictions. Results include
the accepted fraction, accuracy, and bootstrap intervals for the accuracy drop.
Original accuracy is reported both for the full test set and for the accepted
images, counting each original once per accepted variant. Brightness is also
reported as `brightness_dark` (0.4, 0.7) and `brightness_bright` (1.2, 1.5, 1.8).

`results/thresholds/` contains `validation_thresholds.csv`, `test_evaluation.csv`,
`policy_comparison.csv`, `iso_ssim_bins.csv`, and `iso_ssim_curves.png`. The curves
use fixed SSIM bins of width 0.02 from 0.70 to 1.00, keeping bins with at least
50 variants. This is an exploratory analysis added after the initial test results.

## Image degradations

| Degradation | Parameter | Values |
| --- | --- | --- |
| Gaussian blur | Radius in pixels | 0.25, 0.5, 1, 2, 4 |
| JPEG compression | Quality | 90, 70, 50, 30, 10 |
| Reduced resolution | Intermediate side length in pixels | 192, 128, 96, 64, 32 |
| Brightness | Multiplication factor | 0.4, 0.7, 1.2, 1.5, 1.8 |

These values were checked on validation images before being used on test images.
Each variant is made separately from the original resized to 224 x 224 pixels.
For reduced resolution, the image is made smaller and then resized back to
224 x 224. Resizing uses bilinear interpolation. The original files are kept unchanged.

SSIM compares each variant with the resized original before model normalization.
We use it to find different degradations of the same image with similar scores.
The main comparison uses a maximum SSIM difference of 0.01, with 0.02 as an
additional comparison. SSIM measures similarity, not diagnostic correctness.

Reference: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
[Image Quality Assessment: From Error Visibility to Structural Similarity](https://doi.org/10.1109/TIP.2003.819861).

## Report

The Lab 5-8 report is available in [Word](docs/report.docx) and
[Markdown](docs/report.md). It includes the experimental methodology, the case
study on 35 validation images, and related work.

The full test-set results (`results/analysis/`, `results/reliability/`,
`results/bootstrap/`, `results/thresholds/`) were generated as preparation for
validation on the complete dataset. They will be discussed in the next stage of
the report.
