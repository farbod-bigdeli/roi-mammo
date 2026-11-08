# Analysis plan (fixed before any test-set evaluation)

Commit this file, `data/*/split.csv`, `data/*/split_report.txt` and `results/chosen_miniddsm.json`,
then tag the commit (e.g. `git tag prereg-v1`) **before** running the `final` or `extra` plans.
Optionally upload the tagged commit to OSF or Zenodo for a public timestamp.

## Question
Does label-free, saliency-guided ROI cropping during training improve mammogram classification
beyond standard augmentation and beyond random crops of the same size?

## Data and units
- Mini-DDSM, Complete release (Kaggle cheddad/miniddsm2; 7,808 images, 1,952 patients; scanned
  film; cancer vs. benign + normal) and CMMD (TCIA; digital; malignant vs. benign, biopsied
  breasts only).
- Labels at breast level. In Mini-DDSM a breast is positive only if the case is a cancer case
  and that breast carries a lesion outline.
- Mini-DDSM: splits by patient into a locked test set (20 % of patients) and 4 development folds,
  stratified by patient label and site. Split seed 2026.
- CMMD: nothing is tuned on CMMD, so it has no separate test set. All patients are split into
  5 folds (patient-level, stratified by label); final models are evaluated out-of-fold, so every
  patient is predicted exactly once by a model that never saw them (one seed per fold).
- SHA-256 of each `split.csv` is in `split_report.txt`.
- Preprocessing: breast mask (triangle threshold, see README); outside the breast is blacked out.
  Mini-DDSM only: pure-white patches (painted-out film labels, unexposed film edges) are removed
  before segmentation and blacked out, because they occur mostly on scanner A (lowest cancer rate)
  and would be a shortcut. ROI boxes are not scored in a 4 % band along the image edges.
- Breast segmentation checked with `roimammo.qc` (lesion outside the mask, tiny mask, film-edge
  bands, ragged outline, not a breast) after every preparation; flagged rates are reported.
- Duplicate check before splitting (`roimammo.dupcheck`): every patient involved in a
  cross-patient duplicate image (identical prepared pixels, or a near-identical image confirmed
  by eye) is excluded entirely, because the same exam under two IDs would leak through all of
  that patient's images. Excluded patients are listed in `excluded_patients.csv`.

## Selection (development folds only)
- EfficientNet-B0 on Mini-DDSM development folds; ROI grid: box scoring in {blend (texture:
  local variance + LoG energy), intensity (local brightness)} x p in {0.1, 0.25} x K in {1, 5},
  jitter 0.1. Scoring and K are in the grid because the lesion-coverage analysis, run before any
  training, showed that (i) with corrected breast masks the texture score finds lesions no more
  often than randomly scored boxes, while brightness finds them about three times as often as
  chance at K = 1, and (ii) any advantage shrinks as K grows. Coverage numbers used for design
  decisions are computed on development patients only (`coverage --subset dev`); the all-patient
  numbers are reported descriptively.
- The setting with the highest mean breast-level AUC over the 4 folds is chosen and then applied
  unchanged to every dataset and backbone. No tuning on CMMD and no tuning on any test set.

## Training protocol (all arms identical except the augmentation)
- 10 epochs, AdamW (lr 2e-4, weight decay 0.01), cosine schedule with 5 % warm-up, effective
  batch 16, positive-class weight = negatives/positives, input 896 x 448 (letterboxed; breast
  crops are about 2:1).
- The model after the last epoch is evaluated; no checkpoint selection on evaluation data.
- Mini-DDSM: seeds 1, 2, 3 for every final configuration. CMMD: seed 1, one model per fold.

## Arms
Standard; + RandomResizedCrop; + CutMix; + Random crop (same sizes as the bank boxes, random
position in the breast, same p and jitter); + ROI crop (chosen setting).

## Primary endpoint and comparisons
- Breast-level ROC-AUC (breast score = mean of its view probabilities) on the Mini-DDSM locked
  test set and on CMMD out-of-fold predictions.
- Primary comparisons, for every dataset x backbone: (1) ROI crop vs Standard; (2) ROI crop vs
  Random crop.
- Effect = seed-averaged AUC difference. 95 % CI and two-sided p from a paired bootstrap that
  resamples patients (2,000 replicates). Holm correction across all reported comparisons.
- Equivalence: the arms are called equivalent if the 90 % bootstrap CI of the difference lies
  inside +-0.01 AUC (two one-sided tests at alpha = 0.05).

## Secondary
- Patient-level AUC; PR-AUC; Brier score and expected calibration error.
- Paired DeLong test on seed-averaged breast probabilities (ignores the two-breast clustering;
  reported as a check).
- Other comparisons: RandomResizedCrop, CutMix vs Standard; ROI vs RandomResizedCrop.
- Attention-MIL head and two-view (CC + MLO) model, each with and without ROI crops.
- Ablations on development folds, at the chosen scoring, p and K: random-crop control; jitter 0
  and 0.2; the remaining K in {1, 3, 5, 10}; the remaining scoring variants (var, log, random).
- Within-site results for Mini-DDSM (site alone predicts the label with AUC 0.651).
- Mechanism: lesion hit rate and coverage of the top-K boxes vs random boxes of the same sizes
  (Mini-DDSM lesion outlines, analysis only), with patient-bootstrap CIs.
- Site shortcut: cross-validated AUC of predicting the label from site alone.
- Efficiency: steady-state training throughput, data-loading wait, peak allocated GPU memory.

## Reporting
All results are reported whatever their direction, including null and negative results.
Any deviation from this plan is listed in the paper with its reason.
