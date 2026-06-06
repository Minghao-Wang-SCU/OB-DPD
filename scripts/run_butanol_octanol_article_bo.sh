#!/usr/bin/env bash
set -euo pipefail

# Literature-style single-target Bayesian logP fit with balanced water/octanol
# bead counts. The DPD simulation remains constant-pressure; the 1:1 phase
# balance is imposed by the initial molecule counts, not by post-processing.
# solute = 1-butanol [CH3][CH2][CH2][CH2OH]
# organic phase = 1-octanol [CH3][CH2]6[CH2OH]
# water bead uses the current OB-DPD H2O name as the article 2H2O alias.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR_INPUT="${1:-butanol_octanol_article_bo}"
if [[ "${OUTDIR_INPUT}" = /* ]]; then
  OUTDIR="${OUTDIR_INPUT}"
else
  OUTDIR="${ROOT_DIR}/${OUTDIR_INPUT}"
fi
LAMMPS_BIN="${LAMMPS_BIN:-lmp_mpi}"

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

python logp_partition.py fit \
  --solute CH3,CH2,CH2,CH2OH \
  --organic-solvent CH3,CH2,CH2,CH2,CH2,CH2,CH2,CH2OH \
  --allow-custom-beads \
  --target-logp 0.8459 \
  --table "${TABLE}" \
  --fit-pairs H2O:CH3,H2O:CH2,H2O:CH2OH,CH3:CH3,CH3:CH2,CH3:CH2OH,CH2:CH2,CH2:CH2OH,CH2OH:CH2OH \
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
  --bo-full-grid-ei \
  --box 20 20 60 \
  --water-count 17640 \
  --organic-count 2205 \
  --n-solute 180 \
  --steps 1000000 \
  --equilibration-steps 500000 \
  --timestep 0.01 \
  --ensemble nph \
  --pressure 23.7 \
  --pressure-damp 2.0 \
  --dump-every 1000 \
  --analysis-method ummap \
  --analysis-frames 50 \
  --slabs 30 \
  --interface-gradient-fraction 0.3 \
  --interface-slab-padding 1 \
  --angle-param-mode article \
  --job 8	 \
  --lammps "${LAMMPS_BIN}" \
  --outdir "${OUTDIR}" \
  --penalize-failures \
  --gpu
