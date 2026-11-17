#!/usr/bin/env bash
# Checks requested in peer review (CPU only, about 10-20 minutes). Run from the repository root:
#     bash review_checks.sh [CMMD_FOLDER]
# CMMD_FOLDER (optional) is the folder you passed to `prepare --dataset cmmd --root ...`; without it,
# the folder is inferred from data/cmmd/index.csv.
# Everything ends up in review_checks.zip; send that file.
set -euo pipefail
CMMD_ARG=()
[ $# -ge 1 ] && CMMD_ARG=(--cmmd-root "$1")

echo "== 1/4 label audit and CMMD counts"
python -m roimammo.labelaudit --data-root data --out results/labelaudit "${CMMD_ARG[@]}"

echo "== 2/4 main analysis without the breasts with an uncertain label"
python -m roimammo.analyze final --runs runs --exclude results/labelaudit/exclude_breasts.csv \
    --out results/labelaudit/sens_breast | tee results/labelaudit/sens_breast.txt
python -m roimammo.analyze final --study all --runs runs --exclude results/labelaudit/exclude_breasts.csv \
    --out results/labelaudit/sens_breast_all > results/labelaudit/sens_breast_all.txt

echo "== 3/4 replay of the training crop sampler"
python -m roimammo.cropreplay --data-dir data/miniddsm --results results --draws 10 | tee results/cropreplay.txt

echo "== 4/4 within-fold and per-seed CMMD contrasts"
python -m roimammo.foldcheck --runs runs --data-root data --dataset cmmd --study all \
    --out results/foldcheck | tee results/foldcheck.txt

rm -f review_checks.zip
zip -qr review_checks.zip results/labelaudit results/cropreplay results/cropreplay.txt results/foldcheck \
    results/foldcheck.txt -x '*.pdf' '*.png'
[ -f results/cv_runs_miniddsm_effb0.csv ] && zip -q review_checks.zip results/cv_runs_miniddsm_effb0.csv
echo "[done] review_checks.zip"