#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
source "${SCRIPT_DIR}/download_dataset_common.sh"

DOMAIN_SLUG="skin"
DOMAIN_ROOT="$(resolve_domain_root "${DOMAIN_SLUG}")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [DATASET ...]

Domain root:
  ${DOMAIN_ROOT}

Datasets:
  isic2019
  ham10000
  fitzpatrick17k
  ddi
  derm7pt
  all

Environment:
  DOWNLOAD_ROOT  Base root directory. Final path becomes \$DOWNLOAD_ROOT/${DOMAIN_SLUG}

Examples:
  ${SCRIPT_NAME} all
  ${SCRIPT_NAME} isic2019 ham10000
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

download_isic2019() {
  local out="${DOMAIN_ROOT}/ISIC2019"
  local train_input="${out}/ISIC_2019_Training_Input.zip"
  local train_gt="${out}/ISIC_2019_Training_GroundTruth.csv"
  local train_meta="${out}/ISIC_2019_Training_Metadata.csv"
  local test_input="${out}/ISIC_2019_Test_Input.zip"
  local test_gt="${out}/ISIC_2019_Test_GroundTruth.csv"
  local test_meta="${out}/ISIC_2019_Test_Metadata.csv"
  if dataset_done_with_files "$out" "$train_input" "$train_gt" "$train_meta" "$test_input" "$test_gt" "$test_meta"; then
    log "Skipping ISIC2019: marker and files already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading ISIC 2019 bundles to ${out}"
  download_url "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip" "$train_input"
  download_url "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv" "$train_gt"
  download_url "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv" "$train_meta"
  download_url "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip" "$test_input"
  download_url "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv" "$test_gt"
  download_url "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Metadata.csv" "$test_meta"
  mark_dataset_done "$out"
}

download_ham10000() {
  local out="${DOMAIN_ROOT}/HAM10000"
  local part1="${out}/HAM10000_images_part_1.zip"
  local part2="${out}/HAM10000_images_part_2.zip"
  local meta="${out}/HAM10000_metadata.tab"
  local seg="${out}/HAM10000_segmentations_lesion_tschandl.zip"
  local isic_test_gt="${out}/ISIC2018_Task3_Test_GroundTruth.tab"
  local isic_test_img="${out}/ISIC2018_Task3_Test_Images.zip"
  local benefit="${out}/ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.tab"
  if dataset_done_with_files "$out" "$part1" "$part2" "$meta" "$seg" "$isic_test_gt" "$isic_test_img" "$benefit"; then
    log "Skipping HAM10000: marker and files already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading HAM10000 from Harvard Dataverse to ${out}"
  download_dataverse_file 3172585 "$part1"
  download_dataverse_file 3172584 "$part2"
  download_dataverse_file 4338392 "$meta"
  download_dataverse_file 3838943 "$seg"
  download_dataverse_file 6924466 "$isic_test_gt"
  download_dataverse_file 3855824 "$isic_test_img"
  download_dataverse_file 3864681 "$benefit"
  mark_dataset_done "$out"
}

download_fitzpatrick17k() {
  local out="${DOMAIN_ROOT}/Fitzpatrick17k"
  local annotations="${out}/fitzpatrick17k.csv"
  local hf_mirror="${out}/hf_ZYXue_Fitzpatrick_17k"
  if [[ -f "${out}/.done" ]] && is_nonempty_file "$annotations" && dir_has_payload "$hf_mirror"; then
    log "Skipping Fitzpatrick17k: marker, annotations, and HF mirror already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"
  if ! is_nonempty_file "$annotations"; then
    log "Downloading Fitzpatrick17k annotations to ${out}"
    download_url "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv" "$annotations"
  fi
  if have_cmd hf; then
    log "Downloading ZYXue/Fitzpatrick_17k mirror to ${hf_mirror}"
    download_hf_dataset "ZYXue/Fitzpatrick_17k" "$hf_mirror"
  else
    warn "hf CLI unavailable; skipping ZYXue/Fitzpatrick_17k mirror"
  fi
  if is_nonempty_file "$annotations" && dir_has_payload "$hf_mirror"; then
    mark_dataset_done "$out"
  fi
}

download_ddi() {
  local out="${DOMAIN_ROOT}/DDI"
  write_manual_note "$out" \
    "Official source: https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965" \
    "DDI is hosted on the Stanford AIMI portal and currently requires interactive portal access."
}

download_derm7pt() {
  local out="${DOMAIN_ROOT}/Derm7pt"
  if dataset_done_with_dir "$out"; then
    log "Skipping Derm7pt: marker and payload already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"
  if have_cmd kaggle && have_kaggle_auth; then
    log "Downloading Derm7pt via Kaggle mirror to ${out}"
    download_kaggle_dataset "menakamohanakumar/derm7pt" "$out"
    mark_dataset_done "$out"
    return 0
  fi
  warn "Skipping Derm7pt: Kaggle auth is not configured on this host"
}

expand_targets() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      all)
        printf '%s\n' isic2019 ham10000 fitzpatrick17k ddi derm7pt
        ;;
      isic2019|ham10000|fitzpatrick17k|ddi|derm7pt)
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
    isic2019) download_isic2019 ;;
    ham10000) download_ham10000 ;;
    fitzpatrick17k) download_fitzpatrick17k ;;
    ddi) download_ddi ;;
    derm7pt) download_derm7pt ;;
  esac
done

log "Done."
