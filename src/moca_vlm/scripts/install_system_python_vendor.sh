#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PACKAGE_DIR}/requirements-vlm.txt}"
VENDOR_DIR="${VENDOR_DIR:-${PACKAGE_DIR}/vendor/python}"
WHEELHOUSE="${WHEELHOUSE:-}"

mkdir -p "${VENDOR_DIR}"

pip_args=(
  -m pip install
  --upgrade
  --prefer-binary
  --target "${VENDOR_DIR}"
  -r "${REQUIREMENTS_FILE}"
)

if [[ -n "${WHEELHOUSE}" ]]; then
  pip_args+=(--no-index --find-links "${WHEELHOUSE}")
fi

echo "Installing VLM Python dependencies with: ${PYTHON_BIN}"
echo "Target vendor directory: ${VENDOR_DIR}"
"${PYTHON_BIN}" "${pip_args[@]}"

echo
echo "Vendor install complete."
echo "moca_vlm will load dependencies from: ${VENDOR_DIR}"
