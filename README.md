# roi-mammo

Code for a controlled evaluation of label-free, saliency-guided region-of-interest (ROI) cropping
as a training augmentation for mammogram classification, compared with standard augmentation,
RandomResizedCrop, CutMix and size-matched random crops, on Mini-DDSM and CMMD, with three
backbones, three seeds, and a locked patient-level test set.

The analysis is fixed in advance in [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# PyTorch with CUDA from https://pytorch.org/get-started/locally/ , then:
pip install -r requirements.txt
python tests/smoke_test.py                              # ~4 min on CPU, synthetic data
```

All commands run from the repository root. Paths below use `data/`, `runs/`, `results/`.

## 1. Get the data

**Mini-DDSM** (Kaggle, "The Complete Mini-DDSM", cheddad/miniddsm2; ~50 GB). The download must contain
`Data-MoreThanTwoMasks/Data-MoreThanTwoMasks.xlsx` and `MINI-DDSM-Complete-JPEG-8/`
(and optionally `MINI-DDSM-Complete-PNG-16/`). Unzip to e.g. `D:/datasets/miniddsm`.

**CMMD** (The Cancer Imaging Archive). Download the DICOM collection with the NBIA Data Retriever
and the clinical table `CMMD_clinicaldata_revision.xlsx` into one folder, e.g. `D:/datasets/cmmd`.
If the xlsx cannot be read, save it as CSV with "clinical" in the file name.

## 2. Prepare (once per dataset)

```bash
python -m roimammo.prepare --dataset miniddsm --root D:/datasets/miniddsm --out data/miniddsm
python -m roimammo.prepare --dataset cmmd     --root D:/datasets/cmmd     --out data/cmmd
```
Crops every image to the breast (Otsu + morphology, largest component), blacks out labels and
markers outside the breast, downsizes to a long side of at most 1024 px (never upsamples), and
writes `index.csv`. **Copy the printed summary** (class counts, source image sizes): it goes into
the Methods. Mini-DDSM breast labels: a cancer patient's breast without an outline is negative.

## 3. Lock the splits

```bash
python -m roimammo.splits --data-dir data/miniddsm                       # 20 % locked test + 4 folds
python -m roimammo.splits --data-dir data/cmmd --test-frac 0 --folds 5   # out-of-fold evaluation
```
Writes `split.csv` (patient -> test or fold 0-3) and `split_report.txt` (partition sizes, SHA-256,
site x label table, and the AUC of predicting the label from site alone). Commit both files now.

## 4. ROI banks and the mechanism check

```bash
python -m roimammo.roibank --data-dir data/miniddsm --variant blend
for v in var log intensity random; do python -m roimammo.roibank --data-dir data/miniddsm --variant $v --overlays 0; done
python -m roimammo.roibank --data-dir data/cmmd --variant blend
python -m roimammo.roibank --data-dir data/cmmd --variant intensity --overlays 0
python -m roimammo.coverage --data-dir data/miniddsm --subset dev --banks data/miniddsm/roi_bank_*.json   # for decisions
python -m roimammo.coverage --data-dir data/miniddsm --subset all --banks data/miniddsm/roi_bank_*.json   # descriptive
```
**Open `data/miniddsm/roi_overlays_blend.png` and check the boxes sit inside the breast.**
`coverage_summary_<subset>.csv` gives how often the top-K boxes hit the lesion, against random boxes.

## 5. Time one run before planning compute

```bash
python -m roimammo.train --data-dir data/miniddsm --mode cv --fold 0 --arm none --epochs 1 --tag _timing
```
Read `train_img_per_s` and `peak_alloc_mib` in `runs/*_timing/log.csv`, then delete that folder.
On a 4 GB card add `--batch 4 --accum 4` (same effective batch) and, for ConvNeXt or ResNet-50,
`--grad-ckpt 1`. Pass such options to every run of a plan after `--`, as below.

## 6. Development: choose the ROI setting

```bash
python -m roimammo.sweep --plan dev                         # 44 runs (4 folds x 11 configurations)
python -m roimammo.analyze select --dataset miniddsm        # writes results/chosen_miniddsm.json
```
Then **commit ANALYSIS_PLAN.md, the split files and chosen_miniddsm.json, and tag the commit**
(`git tag prereg-v1`) before anything touches the test set.

## 7. Final runs (locked test set) and ablations

```bash
python -m roimammo.sweep --plan final       # 88 runs: 11 arms/backbones x (3 Mini-DDSM seeds + 5 CMMD folds)
python -m roimammo.sweep --plan extra       # 32 runs: attention-MIL head and two-view model, +-ROI
python -m roimammo.sweep --plan ablation    # 36 runs on Mini-DDSM development folds
```
Runs are resumable (finished runs are skipped). To split a plan across machines (e.g. your PC and
Kaggle notebooks), run `--shard 0/2` on one and `--shard 1/2` on the other and copy the `runs/`
folders together. Memory options go after `--`, e.g. `-- --batch 4 --accum 4 --grad-ckpt 1`.
Use `--dry-run` to print the commands.

## 8. Results and figures

```bash
python -m roimammo.analyze final --level breast     # primary
python -m roimammo.analyze final --level patient    # secondary
python -m roimammo.analyze select --dataset miniddsm   # re-run after the ablation plan for the ablation table
python -m roimammo.figures examples --data-dir data/miniddsm --bank data/miniddsm/roi_bank_blend.json
python -m roimammo.figures examples --data-dir data/cmmd --bank data/cmmd/roi_bank_blend.json
python -m roimammo.figures arms --data-dir data/miniddsm --bank data/miniddsm/roi_bank_blend.json
```

| Output (results/) | Use in the paper |
|---|---|
| `forest_breast.pdf` | main result figure |
| `final_summary_breast.csv`, `comparisons_breast.csv` | main tables (AUC, PR-AUC, Brier, ECE; differences, CIs, Holm p, equivalence) |
| `cv_grid_miniddsm_effb0.pdf`, `cv_curves_*.pdf` | ROI-setting selection, learning curves |
| `cv_summary_*.csv` (after ablations) | ablation table |
| `data/miniddsm/coverage_summary_all.csv` | mechanism analysis |
| `fig_examples_*.pdf`, `fig_arms_miniddsm.pdf` | dataset and method figures |
| `efficiency.csv` | throughput / memory table |
| `data/*/split_report.txt` | dataset table, site-shortcut check |

## Design notes (what changed from arXiv v1-v2)

- Labels and units: breast-level labels (a cancer patient's healthy breast is no longer positive);
  patient key = DDSM case (letter + number).
- Locked patient-level test set; ROI setting chosen on development folds of one dataset only and
  transferred unchanged; last-epoch model, never the best validation epoch; three seeds.
- Size-matched random-crop control isolates the effect of where the box is placed.
- ROI boxes keyed by image ID in one shared image frame; training stops with an error if any
  training image has no boxes. Saliency is scored on the breast interior so the skin line does
  not attract boxes. Jitter follows the published equations (independent w, h, centre shifts);
  low-tissue crops are redrawn.
- Breast segmentation: triangle threshold on the image interior, largest component, extension
  into the border band only perpendicular to it (no film-edge strips), hole filling. Checked on
  38 real failure cases (docs/segmentation_before_after.png); run `roimammo.qc` after `prepare`.
- Correct 16-bit handling; letterboxing instead of stretching; ImageNet normalisation.
- Per-sample random streams seeded by (seed, epoch, index): results do not depend on the number
  of data-loader workers.
- Peak memory uses allocated (not reserved) memory; throughput excludes epoch 1 and reports
  data-loading wait.
- Statistics: patient-cluster bootstrap, Holm correction, equivalence margin, DeLong as a check.
