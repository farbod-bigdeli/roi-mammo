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

## Addendum (written 11 Nov 2026, after the main results; fixed before any addendum run)
The main analysis above is final and is not changed by this addendum. Three sensitivity analyses
were added because the main results raised specific questions. Every addendum run is trained with
`--study addendum` and is excluded from the main analysis (`analyze final`, default
`--study main`); the addendum is analysed with `analyze final --study all --out results/addendum`.

**A1. Crop fit.** In the main protocol a roughly square crop is letterboxed into the 896x448
input, leaving about half of the input as zero padding. A1 repeats the two crop arms with the
box widened along its short side to the input aspect ratio (same centre, shifted to stay in the
image; `--crop-fit fill`), after the same tissue check. Backbones: EfficientNet-B0, ResNet-50.
Datasets and seeds as in the main final plan. Comparisons: ROI-fill vs Standard, Random-fill vs
Standard, ROI-fill vs Random-fill (Standard runs reused from the main study). No selection.

**A2. ConvNeXt-T learning rate.** Under the main recipe (lr 2e-4) ConvNeXt-T often stayed on its
initial loss plateau. A2 chooses the ConvNeXt-T learning rate on the Mini-DDSM development folds
only: lr in {2e-4, 1e-4, 5e-5}, standard augmentation, four folds, seed 1, everything else
unchanged. Rule: the highest mean breast-level development AUC (`analyze select-lr`); the final
training loss is reported for every setting. The test set is not used for this choice. With the
chosen lr, ConvNeXt-T is retrained with Standard, Random crop and ROI crop (letterbox), datasets
and seeds as in the main final plan, and compared as in the primary analysis.

**A3. More seeds for CMMD.** The main CMMD results use one seed per fold. A3 adds seeds 2 and 3
for EfficientNet-B0 (all five arms) and ResNet-50 (three arms); the analysis then averages AUC
differences over three seeds, as for Mini-DDSM.

**Analysis.** All addendum comparisons are computed as in the primary analysis (patient bootstrap,
2,000 replicates, equivalence margin +-0.01). Holm correction is applied within the addendum
output, which also re-reports the main comparisons with the added seeds. Addendum results are
reported as sensitivity analyses, separately from the pre-registered results, whatever their
direction.

## Addendum B (written 18 Nov 2026, after the main results, the addendum and the reanalysis; fixed before any run below)
The main analysis and addendum A are final and are not changed by this addendum. Addendum B was
added because a review pointed out that the random-crop control differs from the ROI arm in more
than the saliency: ROI boxes come from windows that need at least 50% breast tissue and are scored
inside the eroded breast, so they also sit more centrally than random crops, which are centred on
any breast pixel and need only 30% tissue. In the reanalysis with 20,000 bootstrap replicates,
ResNet-50 on CMMD with three seeds scored higher with ROI crops than with random crops (+0.010,
Holm p 0.049 within addendum A). Addendum B asks whether that kind of difference comes from the
saliency or from the placement.

**B1. Placement-matched control.** A third crop arm, "central random box", uses the ROI arm unchanged
(p, K, jitter, letterbox, tissue redraw) with the random-score bank (`roi_bank_random.json`,
`roibank --variant random`). That bank uses the same candidate windows, sizes, tissue rule and
non-maximum suppression as the saliency banks; each window is scored by the mean of uniform noise
inside the eroded breast interior (fixed seed per image), so windows lying fully inside the
interior are preferred and ties are broken at random. It shares the placement rules of the ROI
boxes and uses no image content. Runs carry `--study control`. Backbones: EfficientNet-B0 and
ResNet-50. Mini-DDSM: locked test set, seeds 1-3. CMMD: out of fold, seeds 1-3 (5 folds each).
Plan: `sweep --plan control` (36 runs). The Standard, Random crop and ROI crop runs are reused from
the main study and from addendum A3 (CMMD seeds 2-3).

**Comparisons** (per dataset and backbone; 12 in total):
- B-primary: ROI crop vs central random box. This estimates the value of the saliency score with
  the placement held fixed.
- Central random box vs Random crop. This estimates the value of the placement rules.
- Central random box vs Standard.

**Analysis.** `controlarm` computes the paired, seed-averaged breast-level AUC difference with a
patient bootstrap (20,000 replicates), a two-level bootstrap over seeds and patients (5,000
replicates), Holm correction within these 12 comparisons, and equivalence within +-0.01 by a
bootstrap TOST p-value, Holm-adjusted within the same 12. Results are reported whatever their
direction, separately from the pre-specified and addendum A results.

**Interpretation fixed in advance.** For each dataset and backbone, using the 95% patient-bootstrap
intervals: (a) if ROI crop minus central random box lies above zero, the ROI advantage is
attributed to the saliency; (b) if central random box minus Random crop lies above zero and ROI
crop minus central random box includes zero, an ROI advantage over random crops is attributed to
placement; (c) anything else is reported as inconclusive. The case of interest is ResNet-50 on
CMMD; the intervals that also resample seeds are reported alongside.
