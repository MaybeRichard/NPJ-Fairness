#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
source "${SCRIPT_DIR}/download_dataset_common.sh"

DOMAIN_SLUG="physiology"
DOMAIN_ROOT="$(resolve_domain_root "${DOMAIN_SLUG}")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [DATASET ...]

Domain root:
  ${DOMAIN_ROOT}

Datasets:
  ubfc-phys
  bp4dplus
  ecg-fitness
  all

Examples:
  ${SCRIPT_NAME} all
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  usage
  exit 1
fi

mkdir -p "${DOMAIN_ROOT}"

download_ubfc_phys() {
  local out="${DOMAIN_ROOT}/UBFC-PHYS"
  write_manual_note "$out" \
    "Official source: https://sites.google.com/view/ybenezeth/ubfc-phys" \
    "Use the official UBFC-PHYS site for the current dataset link and access instructions."
}

download_bp4dplus() {
  local out="${DOMAIN_ROOT}/BP4D+"
  write_manual_note "$out" \
    "Official source: https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html" \
    "BP4D+ is distributed through the official research page and typically requires an explicit request / agreement."
}

download_ecg_fitness() {
  local out="${DOMAIN_ROOT}/ECG-Fitness"
  write_manual_note "$out" \
    "Official source: https://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/" \
    "The official page provides a request form (ECG-Fitness-request.pdf)." \
    "After the signed request is approved, the maintainers share a 7z download link."
}

expand_targets() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      all)
        printf '%s\n' ubfc-phys bp4dplus ecg-fitness
        ;;
      ubfc-phys|bp4dplus|ecg-fitness)
        printf '%s\n' "$arg"
        ;;
      *)
        die "Unknown dataset target: $arg"
        ;;
    esac
  done
}

mapfile -t TARGETS < <(expand_targets "$@" | awk '!seen[$0]++')

log "Domain root: ${DOMAIN_ROOT}"
for target in "${TARGETS[@]}"; do
  case "$target" in
    ubfc-phys) download_ubfc_phys ;;
    bp4dplus) download_bp4dplus ;;
    ecg-fitness) download_ecg_fitness ;;
  esac
done

log "Done."
