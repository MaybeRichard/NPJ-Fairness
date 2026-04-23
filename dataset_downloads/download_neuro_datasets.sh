#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
source "${SCRIPT_DIR}/download_dataset_common.sh"

DOMAIN_SLUG="neuro"
DOMAIN_ROOT="$(resolve_domain_root "${DOMAIN_SLUG}")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [DATASET ...]

Domain root:
  ${DOMAIN_ROOT}

Datasets:
  openfmri-ds000245
  oasis3
  ppmi
  uk-biobank
  all

Environment:
  DOWNLOAD_ROOT  Base root directory. Final path becomes \$DOWNLOAD_ROOT/${DOMAIN_SLUG}

Examples:
  ${SCRIPT_NAME} all
  ${SCRIPT_NAME} openfmri-ds000245
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

download_openfmri_ds000245() {
  local out="${DOMAIN_ROOT}/OpenfMRI-ds000245"
  local openneuro_cmd=""
  if dataset_done_with_dir "$out"; then
    log "Skipping OpenfMRI ds000245: marker and payload already present"
    return 0
  fi
  if have_cmd openneuro; then
    openneuro_cmd="$(command -v openneuro)"
  elif [[ -x "${HOME}/.local/bin/openneuro-py" ]]; then
    openneuro_cmd="${HOME}/.local/bin/openneuro-py"
  elif have_cmd openneuro-py; then
    openneuro_cmd="$(command -v openneuro-py)"
  fi
  if [[ -z "${openneuro_cmd}" ]]; then
    write_manual_note "$out" \
      "Official dataset page: https://openneuro.org/datasets/ds000245" \
      "Official CLI docs: https://docs.openneuro.org/packages/openneuro-cli.html" \
      "Alternative CLI: pip install --user openneuro-py" \
      "Recommended command: openneuro download ds000245 ${out}" \
      "Alternative command: ~/.local/bin/openneuro-py download --dataset ds000245 --target-dir ${out}"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading OpenNeuro ds000245 into ${out} via ${openneuro_cmd}"
  if [[ "$(basename "${openneuro_cmd}")" == "openneuro" ]]; then
    "${openneuro_cmd}" download ds000245 "$out"
  else
    "${openneuro_cmd}" download --dataset ds000245 --target-dir "$out"
  fi
  mark_dataset_done "$out"
}

download_oasis3() {
  local out="${DOMAIN_ROOT}/OASIS-3"
  write_manual_note "$out" \
    "Official source: https://www.oasis-brains.org/" \
    "OASIS-3 requires registration and data-use agreement acceptance before download."
}

download_ppmi() {
  local out="${DOMAIN_ROOT}/PPMI"
  write_manual_note "$out" \
    "Official source: https://www.ppmi-info.org/access-data-specimens/download-data" \
    "PPMI is portal-driven and requires registration / agreement before export."
}

download_uk_biobank() {
  local out="${DOMAIN_ROOT}/UK-Biobank"
  write_manual_note "$out" \
    "Official source: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access" \
    "UK Biobank requires an approved application and project credentials before data download."
}

expand_targets() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      all)
        printf '%s\n' openfmri-ds000245 oasis3 ppmi uk-biobank
        ;;
      openfmri-ds000245|oasis3|ppmi|uk-biobank)
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
    openfmri-ds000245) download_openfmri_ds000245 ;;
    oasis3) download_oasis3 ;;
    ppmi) download_ppmi ;;
    uk-biobank) download_uk_biobank ;;
  esac
done

log "Done."
