# Experimental Methodology

## 1. Aim of the study

This project studies how image quality affects a skin lesion classifier. The
same images are modified using blur, JPEG compression, reduced resolution, and
brightness changes. The classifier's predictions are then compared with its
predictions on the original images.

The experiments answer three questions:

1. Do different types of degradation lead to different classification results
   when the images have similar quality scores?
2. Which degradation types affect prediction confidence and calibration the
   most at comparable quality levels?
3. Does including the degradation type improve prediction reliability estimation
   compared to using the quality score alone, and does it still help when the
   model's confidence is also considered?

The quality score used here is SSIM.
Experiment 1 provides the initial comparison, Experiment 2 addresses RQ1 and
RQ2, and Experiment 3 addresses RQ3.

The idea being tested is that SSIM may not be enough to describe how a change
in image quality affects the classifier.

## 2. Dataset and splits

### 2.1. HAM10000

The dataset is HAM10000 [1], downloaded from the Kaggle page linked in the
README. It contains 10,015 dermoscopic images of 7,470 lesions, divided into
seven classes.

| Code | Class | Images |
| --- | --- | ---: |
| akiec | Actinic keratoses and intraepithelial carcinoma | 327 |
| bcc | Basal cell carcinoma | 514 |
| bkl | Benign keratosis-like lesions | 1,099 |
| df | Dermatofibroma | 115 |
| mel | Melanoma | 1,113 |
| nv | Melanocytic nevi | 6,705 |
| vasc | Vascular lesions | 142 |

The classes have very different sizes. For this reason, accuracy is reported
alongside balanced accuracy, Macro-F1, and results for each class. A modified
image keeps the diagnosis recorded for its original image.

The dataset-checking script checks the metadata and image filenames for
missing files, duplicate image IDs, unexpected classes, and different diagnoses
assigned to the same lesion.

### 2.2. Separating the data

The ResNet18 model is already trained [2]. The splits are reconstructed using
the author's original code [3], so that evaluation follows the documented split.
Lesion IDs are sorted, shuffled with seed 42, and divided into 70% training,
15% validation, and the remaining 15% test lesions. The first two counts are
rounded down.

| Split | Images | Lesions | Use in this project |
| --- | ---: | ---: | --- |
| Training | 6,986 | 5,229 | Part of the downloaded model's documented split; not used for retraining |
| Validation | 1,521 | 1,120 | Exploring degradations and fitting reliability estimators |
| Test | 1,508 | 1,121 | Evaluating the classifier and reliability estimators |

All photographs of one lesion stay in the same split. The script checks that
no lesion appears in two splits and saves the assignments in `data/splits.csv`.
The image counts and test class counts match the author's records. This checks
the reconstructed split; the weights alone cannot prove which images were used
in training. Separation is checked by lesion, rather than by patient. If
$\mathcal L_{train}$, $\mathcal L_{val}$, and $\mathcal L_{test}$ are the sets
of lesion IDs, the required condition is

$$
\mathcal L_{train}\cap\mathcal L_{val}=\varnothing,\qquad
\mathcal L_{train}\cap\mathcal L_{test}=\varnothing,\qquad
\mathcal L_{val}\cap\mathcal L_{test}=\varnothing.
$$

## 3. Model and image preparation

The classifier is ResNet18 with a final layer for the seven HAM10000 classes
[2]. The class order is `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`.
Its downloaded weights are kept fixed throughout the experiments.

The weights are `resnet.pth` from revision
`645de358497899dbed2e4ca3ed137305404131f7`. Their SHA-256 checksum is
`0b90451a8727dda76df5664bd48faf7228d5036eddf147da1df9bd282c1371b5`.

Each image is converted to RGB and resized to $224\times224$ pixels using bilinear
interpolation, as required by the model. Degradations and SSIM are calculated
at this size, before normalization. For the classifier, pixel values are divided
by 255 and normalized using ImageNet means `(0.485, 0.456, 0.406)` and standard
deviations `(0.229, 0.224, 0.225)`. For RGB channel $h$:

$$
\widetilde x_h=\frac{x_h/255-\mu_h}{\sigma_h},\qquad
\mu=(0.485,0.456,0.406),\quad \sigma=(0.229,0.224,0.225).
$$

The model runs on CPU in evaluation mode, without computing gradients, using
batches of 16 and four PyTorch threads. The predicted class is the class with
the highest softmax probability. That probability is called **confidence**.
For example, confidence 0.9 means the model assigned 90% probability to its
chosen class; it does not guarantee that 90% of such predictions are correct.

For an input image $v$, let $z_k(v)$ be the model output before softmax for
class $k$. Probabilities, predicted class, and confidence are

$$
p_k(v)=\frac{\exp(z_k(v))}{\sum_{j=1}^{7}\exp(z_j(v))},\qquad
\widehat y(v)=\arg\max_k p_k(v),\qquad c(v)=\max_k p_k(v).
$$

Here $x_i$ is the resized original and $y_i$ its recorded class. Correctness is

$$
t_i(v)=\mathbf 1\{\widehat y(v)=y_i\}.
$$

The indicator $\mathbf 1$ equals 1 when the prediction is correct and 0 otherwise.
For original images, write $t_i=t_i(x_i)$ and $c_i=c(x_i)$. For the variants
defined below, write $t_{i,d,a}=t_i(x_{i,d,a})$ and
$c_{i,d,a}=c(x_{i,d,a})$. In metric formulas, $t_i$ and $c_i$ also denote the
correctness and confidence of row $i$ in the condition being evaluated.

## 4. Degradations and SSIM

### 4.1. Creating the modified images

Each variant is made directly from the resized original. The transformations
are applied separately, so a blurred variant is not also JPEG-compressed or
darkened. Original files are kept unchanged.

A variant can be written as

$$
x_{i,d,a}=T_{d,a}(x_i),
$$

where $d$ is the degradation type and $a$ is its parameter.

| Type | Operation | Parameters |
| --- | --- | --- |
| Blur | Pillow Gaussian blur; radius is the Gaussian standard deviation in pixels | Radius: 0.25, 0.5, 1, 2, 4 |
| JPEG | Encode and decode the image, using 4:2:0 chroma subsampling | Quality: 90, 70, 50, 30, 10 |
| Reduced resolution | Resize to a smaller square, then back to $224\times224$, using bilinear interpolation | Side length: 192, 128, 96, 64, 32 pixels |
| Brightness | Multiply pixel intensities by a factor | Factor: 0.4, 0.7, 1.2, 1.5, 1.8 |

There are 20 variants per image. Brightness factors below 1 darken the image;
factors above 1 brighten it. Values above the pixel range are clipped. JPEG
adds compression to an image that has already been decoded and resized.
Pillow approximates Gaussian filtering. Brightness with factor $b$ is

$$
T_{brightness,b}(x)=\operatorname{clip}(bx,0,255),
$$

with conversion to integer pixel values in the implementation.

### 4.2. SSIM calculation

SSIM [4] compares the structure, contrast, and brightness of two images. Here,
it compares each variant with its resized original:

$$
q_{i,d,a}=SSIM(x_i,x_{i,d,a}).
$$

For one local window, the SSIM formula used with its standard settings is

$$
SSIM(x,v)=
\frac{(2\mu_x\mu_v+C_1)(2\sigma_{xv}+C_2)}
{(\mu_x^2+\mu_v^2+C_1)(\sigma_x^2+\sigma_v^2+C_2)}.
$$

$\mu_x$ and $\mu_v$ are local means, $\sigma_x^2$ and $\sigma_v^2$ are local
variances, and $\sigma_{xv}$ is their covariance. The first ratio compares
luminance; the second combines contrast and structure. The constants are
$C_1=(0.01R)^2$ and $C_2=(0.03R)^2$, with pixel range $R=255$.
Local scores are averaged over image positions and RGB channels.

Identical images have SSIM 1. Lower scores mean lower similarity according to
this metric. SSIM needs the original as a reference; it does not measure medical
usefulness directly, and the original itself may have quality problems.

The implementation uses `structural_similarity` from scikit-image with
`data_range=255`, `channel_axis=-1`, `gaussian_weights=True`, `sigma=1.5`, and
`use_sample_covariance=False`.

Since SSIM includes a luminance term, brightness changes affect the score even
without blur or loss of resolution. Comparisons involving brightness therefore
need to account for what this similarity score measures; a similar SSIM is not
a guarantee that two variants preserve the same useful image details.

### 4.3. Initial check on validation images

The initial check uses 35 validation images: five from each class, all from
different lesions. Images are sorted by ID, one per lesion is kept, and five
per class are sampled with seed 42. Their IDs are saved in
`data/validation_selection.csv`.

The script creates 700 variants and records their SSIM scores, without running
the classifier. It also checks how often variants of different types can be
matched within SSIM differences of 0.01 and 0.02. The parameter grid and main
tolerance of 0.01 were fixed before running the degraded test images. A second
tolerance of 0.02 is used to check whether the matching results depend on this
choice. The initial check covers 35 validation images.

## 5. Experiments

### 5.1. Experiment 1: classification before and after degradation

ResNet is run on the 1,508 original test images and their 30,160 variants.
Each output row records the image and lesion IDs, true class, predicted class,
confidence, and class probabilities. For variants, it also records degradation
type, parameter, and SSIM.

Metrics are calculated for originals and for each of the 20 conditions. Each
condition contains the same source images, making the comparisons consistent.
Differences from the originals are descriptive: no confidence intervals have
been calculated for these Experiment 1 differences.

### 5.2. Experiment 2: comparing variants with similar SSIM

For each source image and each pair of degradation types, all 25 combinations
of parameter values are checked. The chosen combination has the smallest
absolute SSIM difference:

$$
(a_i^*,b_i^*)\in\arg\min_{a\in\mathcal A_d,\ b\in\mathcal A_e}
|q_{i,d,a}-q_{i,e,b}|,\qquad
\Delta q_i=|q_{i,d,a_i^*}-q_{i,e,b_i^*}|.
$$

Here $\mathcal A_d$ and $\mathcal A_e$ are the parameter sets in Section 4.1.
The pair is kept only if $\Delta q_i\leq\varepsilon$, where the main tolerance
is $\varepsilon=0.01$. If several pairs have the same difference, the smaller
numeric parameter values are chosen, first for the first type and then for
the second. Predictions and confidence are not used to select pairs.

All six pairs of degradation types are compared. Each source image contributes
at most one pair per comparison. The analysis is repeated with tolerance 0.02.
For each comparison, the script reports the number of matched images and
lesions, SSIM gaps, accuracy, mean confidence, ECE, and how often only one of
the two predictions is correct.

Let $\mathcal M$ contain the accepted image IDs and $n=|\mathcal M|$.
For these matched pairs, the accuracy difference is

$$
\Delta Acc=\frac{1}{n}\sum_{i\in\mathcal M}(t_{i,d,a_i^*}-t_{i,e,b_i^*}).
$$

A positive value means the first type has higher accuracy on these pairs.
Confidence differences use the same subtraction. Each type comparison can
contain different images, since an image may have a match for some types but
not others. Similar SSIM does not guarantee equal diagnostic information.

### 5.3. Experiment 3: estimating whether ResNet is correct

This experiment uses logistic regression to estimate the probability that a
ResNet prediction is correct. It does not predict the lesion class or retrain
ResNet. It is fitted on the 30,420 degraded validation rows and evaluated on
the 30,160 degraded test rows.

| Estimator | Inputs |
| --- | --- |
| A | SSIM |
| B | SSIM and degradation type |
| C | SSIM and ResNet confidence |
| D | SSIM, degradation type, and ResNet confidence |
| E | ResNet confidence |

For input features $\phi$, the estimated probability is

$$
\widehat r=\frac{1}{1+\exp[-(\beta_0+\beta^T\phi)]}.
$$

The coefficients $\beta$ are learned from the validation data, using correctness
$t$ as the target. Numeric inputs are standardized using validation means and
standard deviations. Degradation types are encoded as separate binary columns,
with the first category omitted. Image IDs, lesion IDs, and class labels are
not input features.

All estimators use the same settings: `C=1`, `solver="lbfgs"`, `max_iter=1000`,
`tol=1e-6`, and `random_state=42`, with L2 regularization and no class weighting.
There is no parameter search on the test set.

B versus A tests whether type adds information beyond SSIM. D versus C tests
whether it still helps when confidence is included. E provides a comparison
using confidence alone. The original model author also used the validation
split for model selection; here it is used for fitting the estimators, rather
than for reporting their final performance.

## 6. Evaluation metrics

### 6.1. Classification metrics

Accuracy is the fraction of correct predictions:

$$
Acc=\frac{1}{n}\sum_{i=1}^{n}t_i.
$$

Balanced accuracy is the average recall across the seven classes. Macro-F1 is
the average of their F1 scores, giving each class equal weight. For class $k$:

$$
P_k=\frac{TP_k}{TP_k+FP_k},\qquad Recall_k=\frac{TP_k}{TP_k+FN_k},\qquad
F1_k=\frac{2TP_k}{2TP_k+FP_k+FN_k}.
$$

$P_k$ is precision. $TP$, $FP$, and $FN$ are true positives, false positives, and false negatives
for that class. Undefined precision or F1 is set to zero. Mean confidence is
the average of the highest class probabilities.

### 6.2. Calibration

Calibration describes how closely confidence agrees with actual accuracy.
ECE [5] groups predictions into ten confidence bins: $[0,0.1)$ through
$[0.9,1]$. It then averages the gap between accuracy and confidence:

$$
ECE=\sum_{j=1}^{10}\frac{|B_j|}{n}
|Acc(B_j)-Conf(B_j)|.
$$

$B_j$ is one bin, and empty bins contribute zero. Lower ECE means smaller gaps
for this choice of bins. A calibration plot and bin counts are also saved for
the original images.

### 6.3. Reliability metrics

The estimators are evaluated using error AUROC, Average Precision (AP), and
Brier score. Error is the positive class: the target is $1-t_i$, and the error
score is $1-\widehat r_i$.

AUROC measures how well errors are ranked above correct predictions. AP
summarizes precision and recall for error detection; the error rate is also
reported as a reference. Higher AUROC and AP are better.

Brier score measures the difference between estimated correctness and the
actual outcome:

$$
BS=\frac{1}{n}\sum_{i=1}^{n}(\widehat r_i-t_i)^2.
$$

Lower Brier is better. The metrics cover degraded image variants, which are
not independent patients.

## 7. Confidence intervals

Images and variants from the same lesion are related. Bootstrap sampling
therefore selects whole lesions, keeping their rows together. In each
repetition, lesion IDs are sampled with replacement. If a lesion is selected
several times, all its rows receive that same repetition count as their weight.
Both sides of a comparison use the same sampled lesions. With $L$ distinct
lesions, one repetition draws $\ell_1^*,\ldots,\ell_L^*$ with replacement.
The weight of lesion $\ell$ is

$$
w_\ell=\sum_{j=1}^{L}\mathbf 1\{\ell_j^*=\ell\}.
$$

Every row from that lesion receives weight $w_\ell$. For example, the bootstrap
accuracy difference for matched pairs is

$$
\Delta Acc^*=\frac{\sum_{i\in\mathcal M}w_{\ell(i)}
(t_{i,d,a_i^*}-t_{i,e,b_i^*})}{\sum_{i\in\mathcal M}w_{\ell(i)}}.
$$

Here $\ell(i)$ is the lesion of image $i$. The other bootstrap metrics use the
same row weights.

There are 1,000 repetitions with seed 42. The 2.5th and 97.5th percentiles of
the differences give a 95% confidence interval. Intervals are calculated for:

- Accuracy and mean-confidence differences in Experiment 2, at both tolerances.
- AUROC, AP, and Brier differences for B minus A and D minus C in Experiment 3.

For AUROC and AP, positive differences favor adding the type. For Brier,
negative differences favor it. Ranking metrics require both correct and
incorrect predictions in a resample. The valid repetition count is recorded;
all 1,000 repetitions were valid in the completed reliability analysis.

The estimators stay fixed during bootstrap: they are not refitted. These
intervals describe variation in the test sample, without adjusting for the
number of comparisons. A difference can be consistent across resamples while
still being too small to matter in practice.

### 7.1. How the research questions are evaluated

For RQ1, the main evidence is the matched accuracy difference at tolerance
0.01 and its 95% interval. An interval excluding zero supports a difference
for that type pair; tolerance 0.02 checks sensitivity to the matching rule.
For RQ2, mean-confidence differences have bootstrap intervals, while matched
ECE values are descriptive because intervals for ECE differences have not
been calculated. The comparisons do not establish one universal ranking of
types, since each pair may contain different matched images.

For RQ3, B minus A and D minus C measure the contribution of type. An AUROC
interval entirely above zero supports improved error ranking for that
comparison. AP and Brier provide additional checks of detection performance
and probability estimates.

This interpretation is exploratory and was defined after examining the initial
test results. Practical relevance is assessed using the size of the differences,
retained accuracy, and the fraction of errors removed. No minimum practical
effect was specified before evaluation.

## 8. Additional analyses

The confidence-only estimator E, deferral analysis, and class-level analyses
were added after the initial test results were examined. They are exploratory
analyses, rather than confirmation on new independent data.

### 8.1. Setting aside uncertain predictions

Predictions are ranked by estimated correctness. The lowest 10% or 20% are
set aside, and accuracy is measured on the remaining predictions. The 0% case
keeps everything. The number set aside is `round(fraction * number_of_rows)`.
Ties use a random order with seed 42, without looking at correctness.

Let $\mathcal K$ be the retained rows and $\mathcal J$ the rows set aside,
from $n$ total prediction rows. The measures are

$$
Coverage=\frac{|\mathcal K|}{n},\qquad
Acc_{retained}=\frac{\sum_{i\in\mathcal K}t_i}{|\mathcal K|},
$$

$$
ErrorFraction_{removed}=\frac{\sum_{i\in\mathcal J}(1-t_i)}
{\sum_{i=1}^{n}(1-t_i)}.
$$

The last fraction is undefined if there are no errors. The outputs include
these measures and the numbers of predictions and errors set aside, following
selective classification [10].

### 8.2. Results by class

Classification and reliability metrics are also calculated for each diagnosis.
The deferral rule is applied globally, so the tables show how many predictions
from each class remain. This helps check whether improvements mainly benefit
some classes or come with more rejection of others.

AUROC is left undefined if a class has only correct or only incorrect outcomes.
AP is left undefined if it has no errors. Small classes need careful reading;
for example, the test split contains only six `df` lesions. No bootstrap
intervals have been calculated for these class tables, deferral, or E comparisons.

## 9. Link to previous studies

Earlier studies examine how image artefacts affect dermatology classifiers
[6, 7]. DermaIQA [8] focuses on assessing dermoscopic image quality. Drenkow
and Unberath [9] study the relationship between quality metrics and classifier
correctness, including logistic regression. DACOM [11] combines quality and
distortion information for a different task: AI-generated-image detection.

This project studies these questions on HAM10000 with a fixed ResNet18. It
compares variants of the same image at similar SSIM and tests whether the
degradation type adds useful information to simple correctness estimators.
These experiments do not introduce a new architecture or quality metric.

A and C are the main baselines for checking the contribution of type, with E
as an additional confidence-only comparison. A represents estimation based
only on the quality score. The confidence input is Maximum Softmax Probability
(MSP), used by Hendrycks and Gimpel [12] as a baseline for detecting incorrect
predictions. E fits logistic regression on MSP to estimate correctness; it is
an MSP-based baseline, rather than a direct evaluation of the raw MSP score.
In particular, its Brier score evaluates the fitted probability estimates.
Any improvement concerns estimating correctness or detecting errors. It does not improve ResNet's predictions,
which remain unchanged. Results from different datasets and models cannot be
compared directly as proof that one method is better.

## 10. Implementation and limitations

The project uses Python 3.12 and PyTorch for ResNet, Pillow for image changes,
scikit-image for SSIM, scikit-learn for metrics and logistic regression, pandas
and NumPy for tables and calculations, and Matplotlib for figures. PyTorch was
chosen because the downloaded model uses it. Package versions are recorded in
`requirements.txt`.

TensorFlow is an alternative for neural-network implementation, but would
require adapting the PyTorch model. OpenCV could provide image transformations;
Pillow was sufficient for the four operations used here. The `piq` library is
an alternative source of image-quality metrics, but scikit-image already
provides the selected SSIM calculation.

Splits, predictions, metrics, and confidence intervals are saved as CSV files.
Code and documentation are stored in Git. Dataset images and model weights
are excluded; the README gives their download sources and execution commands.

The main limitations are using one dataset, one classifier, one split, and a
small fixed set of synthetic degradations. SSIM requires an original reference
image. The degradation type is known because the code creates it, whereas a
real image can contain several unknown quality problems. Logistic regression
also may not capture more complex relationships between the inputs.

The results apply to the dataset, classifier, and degradation conditions
evaluated in this project.

## References

[1] Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset,
a large collection of multi-source dermatoscopic images of common pigmented
skin lesions. *Scientific Data*, 5, 180161.
https://doi.org/10.1038/sdata.2018.161

[2] miloszkrawczyk. Skin lesion classifier (HAM10000). Public model card and weights.
https://huggingface.co/miloszkrawczyk/skin-lesion-classifier

[3] miloszkrawczyk. Melanoma classifier: splitting code and ResNet18 training record,
revision `b3ed99dc4b370ea97f799f096e0c148bf5d82bdd`.
[Code](https://github.com/miloszkrawczyk/melanoma-classifier/blob/b3ed99dc4b370ea97f799f096e0c148bf5d82bdd/src/data.py),
[training record](https://github.com/miloszkrawczyk/melanoma-classifier/blob/b3ed99dc4b370ea97f799f096e0c148bf5d82bdd/results/04-resnet18.txt).

[4] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
Image quality assessment: From error visibility to structural similarity.
*IEEE Transactions on Image Processing*, 13(4), 600-612.
https://doi.org/10.1109/TIP.2003.819861

[5] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
On calibration of modern neural networks. *Proceedings of ICML*, 70, 1321-1330.
https://proceedings.mlr.press/v70/guo17a.html

[6] Akkoca Gazioglu, B. S., & Kamasak, M. E. (2021). Effects of objects and image
quality on melanoma classification using deep neural networks.
*Biomedical Signal Processing and Control*, 67, 102530.
https://doi.org/10.1016/j.bspc.2021.102530

[7] Petrie, T. C., et al. (2021). Quantifying acceptable artefact ranges for
dermatologic classification algorithms. *Skin Health and Disease*, 1(2), e19.
https://doi.org/10.1002/ski2.19

[8] Ferraris, A., et al. (2026). No-reference quality assessment of dermoscopic
images using minimal expert supervision. *Applied Sciences*, 16(4), 1682.
https://doi.org/10.3390/app16041682

[9] Drenkow, N., & Unberath, M. (2025). A causal framework for aligning image
quality metrics and deep neural network robustness.
https://arxiv.org/abs/2503.02797

[10] Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural
networks. https://arxiv.org/abs/1705.08500

[11] Li, B., Li, H., Li, H., Zhong, J., Chen, C., Ni, J., & Cao, B. (2026).
Enabling your forensic detector know how well it performs on distorted samples.
*ICLR 2026*. https://openreview.net/forum?id=Jz5SA2KoFt

[12] Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified
and out-of-distribution examples in neural networks. *ICLR 2017*.
https://arxiv.org/abs/1610.02136
