#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${OBDPD_ENV_NAME:-dpd}"
LAMMPS_BIN="${LAMMPS_BIN:-lmp_mpi}"
PACKMOL_BIN="${PACKMOL_BIN:-packmol}"

usage() {
  cat <<EOF
Usage: bash scripts/install_env.sh [options]

Options:
  -n, --name NAME          Conda environment name (default: dpd)
      --lammps-bin PATH    LAMMPS executable for validation (default: lmp_mpi)
      --packmol-bin PATH   Packmol executable for validation (default: packmol)
  -h, --help               Show this help

This script installs the Python environment from environment.yml and registers
the repository-local auto_mapping source path through Conda activation hooks.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
      ENV_NAME="$2"
      shift 2
      ;;
    --lammps-bin)
      LAMMPS_BIN="$2"
      shift 2
      ;;
    --packmol-bin)
      PACKMOL_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -d "${ROOT_DIR}/auto_mapping" ]]; then
  echo "ERROR: auto_mapping directory not found under ${ROOT_DIR}" >&2
  exit 1
fi

echo "[OB-DPD] repository: ${ROOT_DIR}"
echo "[OB-DPD] conda environment: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[OB-DPD] updating existing environment from environment.yml"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[OB-DPD] creating environment from environment.yml"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/ob-dpd.sh" <<EOF
export OB_DPD_HOME="${ROOT_DIR}"
export OB_DPD_AUTO_MAPPING="${ROOT_DIR}/auto_mapping"
export OB_DPD_AUTO_MAPPING_M3="${ROOT_DIR}/auto_mapping/cg_param_m3-martini3_v2"
export OB_DPD_OLD_PYTHONPATH="\${PYTHONPATH:-}"
if [ -n "\${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/auto_mapping:${ROOT_DIR}/auto_mapping/cg_param_m3-martini3_v2:\${PYTHONPATH}"
else
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/auto_mapping:${ROOT_DIR}/auto_mapping/cg_param_m3-martini3_v2"
fi
EOF

cat > "${DEACTIVATE_DIR}/ob-dpd.sh" <<'EOF'
if [ -n "${OB_DPD_OLD_PYTHONPATH+x}" ]; then
  export PYTHONPATH="${OB_DPD_OLD_PYTHONPATH}"
else
  unset PYTHONPATH
fi
unset OB_DPD_HOME
unset OB_DPD_AUTO_MAPPING
unset OB_DPD_AUTO_MAPPING_M3
unset OB_DPD_OLD_PYTHONPATH
EOF

echo "[OB-DPD] activation hooks installed in ${ENV_PREFIX}/etc/conda"

conda run -n "${ENV_NAME}" python "${ROOT_DIR}/scripts/check_install.py" \
  --root "${ROOT_DIR}" \
  --lammps-bin "${LAMMPS_BIN}" \
  --packmol-bin "${PACKMOL_BIN}"

echo
echo "[OB-DPD] installation complete."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
