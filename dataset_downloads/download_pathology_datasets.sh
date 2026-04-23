#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
source "${SCRIPT_DIR}/download_dataset_common.sh"

DOMAIN_SLUG="pathology"
DOMAIN_ROOT="$(resolve_domain_root "${DOMAIN_SLUG}")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [DATASET ...]

Domain root:
  ${DOMAIN_ROOT}

Datasets:
  tcga
  tcga-brca
  tcga-ucec
  tcga-crc
  cptac-ucec
  camelyon17
  abctb
  all

Environment:
  DOWNLOAD_ROOT         Base root directory. Final path becomes \$DOWNLOAD_ROOT/${DOMAIN_SLUG}
  TCGA_MANIFEST         Generic GDC manifest path for the tcga target.
  TCGA_BRCA_MANIFEST    GDC manifest path for TCGA-BRCA.
  TCGA_UCEC_MANIFEST    GDC manifest path for TCGA-UCEC.
  TCGA_CRC_MANIFEST     GDC manifest path for CRC-related projects, typically TCGA-COAD + TCGA-READ.
  GDC_TOKEN_FILE        Optional GDC token file if your manifest includes controlled data.

Examples:
  ${SCRIPT_NAME} camelyon17
  TCGA_BRCA_MANIFEST=/path/to/gdc_manifest.txt ${SCRIPT_NAME} tcga-brca
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

download_gdc_manifest_bundle() {
  local out="$1"
  local manifest_var="$2"
  local project_hint="$3"
  local portal_url="$4"

  if dataset_done_with_dir "$out"; then
    log "Skipping $(basename "$out"): marker and payload already present"
    return 0
  fi

  local manifest_path="${!manifest_var:-}"
  if ! have_cmd gdc-client || [[ -z "$manifest_path" || ! -f "$manifest_path" ]]; then
    write_manual_note "$out" \
      "Official source: ${portal_url}" \
      "Generate or export a GDC manifest for ${project_hint} from the official portal." \
      "Set ${manifest_var}=/path/to/gdc_manifest.txt and rerun this target." \
      "Optional: set GDC_TOKEN_FILE if the manifest includes controlled files."
    return 0
  fi

  mkdir -p "$out"
  log "Downloading ${project_hint} via gdc-client into ${out}"
  local cmd=(gdc-client download -m "$manifest_path" -d "$out")
  if [[ -n "${GDC_TOKEN_FILE:-}" ]]; then
    cmd+=(--token-file "$GDC_TOKEN_FILE")
  fi
  "${cmd[@]}"
  mark_dataset_done "$out"
}

download_tcga() {
  download_gdc_manifest_bundle \
    "${DOMAIN_ROOT}/TCGA" \
    "TCGA_MANIFEST" \
    "the selected TCGA project(s)" \
    "https://portal.gdc.cancer.gov/"
}

download_tcga_brca() {
  download_gdc_manifest_bundle \
    "${DOMAIN_ROOT}/TCGA-BRCA" \
    "TCGA_BRCA_MANIFEST" \
    "TCGA-BRCA" \
    "https://portal.gdc.cancer.gov/projects/TCGA-BRCA"
}

download_tcga_ucec() {
  download_gdc_manifest_bundle \
    "${DOMAIN_ROOT}/TCGA-UCEC" \
    "TCGA_UCEC_MANIFEST" \
    "TCGA-UCEC" \
    "https://portal.gdc.cancer.gov/projects/TCGA-UCEC"
}

download_tcga_crc() {
  download_gdc_manifest_bundle \
    "${DOMAIN_ROOT}/TCGA-CRC" \
    "TCGA_CRC_MANIFEST" \
    "TCGA colorectal cohorts, typically TCGA-COAD and TCGA-READ" \
    "https://portal.gdc.cancer.gov/"
}

download_cptac_ucec() {
  local out="${DOMAIN_ROOT}/CPTAC-UCEC"
  write_manual_note "$out" \
    "Official source: https://proteomic.datacommons.cancer.gov/pdc/" \
    "CPTAC-UCEC data are typically distributed through the Proteomic Data Commons / CPTAC portal and require portal-driven export."
}

download_camelyon17() {
  local out="${DOMAIN_ROOT}/CAMELYON17"
  if dataset_done_with_dir "$out"; then
    log "Skipping CAMELYON17: marker and payload already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"
  if have_cmd kaggle && have_kaggle_auth; then
    log "Downloading CAMELYON17 via Kaggle mirror to ${out}"
    download_kaggle_dataset "mahdibonab/camelyon17" "$out"
    mark_dataset_done "$out"
    return 0
  fi
  if have_cmd hf; then
    log "Downloading CAMELYON17 via Hugging Face mirror to ${out}"
    download_hf_dataset "jxie/camelyon17" "$out"
    mark_dataset_done "$out"
    return 0
  fi
  warn "Skipping CAMELYON17: neither Kaggle auth nor hf CLI is available"
}

download_abctb() {
  local out="${DOMAIN_ROOT}/ABCTB"
  write_manual_note "$out" \
    "Official source: https://www.abctb.org.au/" \
    "ABCTB access is cohort / portal based and typically requires application or study-specific approval."
}

expand_targets() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      all)
        printf '%s\n' tcga tcga-brca tcga-ucec tcga-crc cptac-ucec camelyon17 abctb
        ;;
      tcga|tcga-brca|tcga-ucec|tcga-crc|cptac-ucec|camelyon17|abctb)
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
    tcga) download_tcga ;;
    tcga-brca) download_tcga_brca ;;
    tcga-ucec) download_tcga_ucec ;;
    tcga-crc) download_tcga_crc ;;
    cptac-ucec) download_cptac_ucec ;;
    camelyon17) download_camelyon17 ;;
    abctb) download_abctb ;;
  esac
done

log "Done."
