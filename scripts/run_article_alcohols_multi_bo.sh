#!/usr/bin/env bash
set -euo pipefail

# Multi-target Bayesian logP fitting aligned with the article-style CH2 bead
# scheme. Each BO evaluation runs four independent partition simulations:
# 1-propanol, 1-butanol, 1-pentanol, and 1,4-butanediol in octanol/water.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR_INPUT="${1:-article_alcohols_multi_logp_bo}"
if [[ "${OUTDIR_INPUT}" = /* ]]; then
  OUTDIR="${OUTDIR_INPUT}"
else
  OUTDIR="${ROOT_DIR}/${OUTDIR_INPUT}"
fi
LAMMPS_BIN="${LAMMPS_BIN:-lmp_mpi}"
BO_ACQ_CANDIDATES="${BO_ACQ_CANDIDATES:-5000}"
BO_EI_ARGS=(--no-bo-full-grid-ei --bo-acq-candidates "${BO_ACQ_CANDIDATES}")
if [[ "${BO_FULL_GRID_EI:-0}" == "1" ]]; then
  echo "[OB-DPD] BO EI mode: full discrete grid"
  BO_EI_ARGS=(--bo-full-grid-ei)
else
  echo "[OB-DPD] BO EI mode: random candidates (${BO_ACQ_CANDIDATES})"
fi

mkdir -p "${OUTDIR}"
TABLE="${OUTDIR}/anderson_ch2_table.csv"

cat > "${TABLE}" <<'CSV'
beadi,beadj,A_ij,R_ij,r0
H2O,H2O,25.0,1.0000,
H2O,CH3,35.0,0.9785,
H2O,CH2,39.0,0.9625,
H2O,CH2OH,14.0,0.9900,
CH3,CH3,36.0,0.9570,
CH3,CH2,19.0,0.9410,0.30
CH3,CH2OH,53.0,0.9525,0.35
CH2,CH2,21.0,0.9250,0.30
CH2,CH2OH,22.0,0.9370,0.35
CH2OH,CH2OH,31.0,0.9800,
CSV

cd "${ROOT_DIR}"

python logp_partition.py fit-staged \
  --stage-config examples/article_alcohols_multi_logp_bo.json \
  --table "${TABLE}" \
  --optimizer bayesian \
  --bo-initial 10 \
  --max-iter 40 \
  --bo-grid H2O:CH3=35:47:2 \
  --bo-grid H2O:CH2=39:51:2 \
  --bo-grid H2O:CH2OH=14:26:2 \
  --bo-grid CH3:CH3=24:36:2 \
  --bo-grid CH3:CH2=19:31:2 \
  --bo-grid CH3:CH2OH=41:53:2 \
  --bo-grid CH2:CH2=9:21:2 \
  --bo-grid CH2:CH2OH=22:34:2 \
  --bo-grid CH2OH:CH2OH=25:37:2 \
  "${BO_EI_ARGS[@]}" \
  --job 8 \
  --lammps "${LAMMPS_BIN}" \
  --outdir "${OUTDIR}" \
  --penalize-failures \
  --gpu
