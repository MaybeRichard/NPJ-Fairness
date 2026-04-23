#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
source "${SCRIPT_DIR}/download_dataset_common.sh"

DOMAIN_SLUG="chest"
DOMAIN_ROOT="$(resolve_domain_root "${DOMAIN_SLUG}")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [DATASET ...]

Domain root:
  ${DOMAIN_ROOT}

Datasets:
  mimic-cxr-jpg
  chexpert
  chestxray14
  montgomery
  shenzhen
  brixia-covid19
  bimcv-covid19
  tb-portals
  rsna-pneumonia
  midrc
  fairfedmed-chest
  all

Environment:
  DOWNLOAD_ROOT         Base root directory. Final path becomes \$DOWNLOAD_ROOT/${DOMAIN_SLUG}
  PHYSIONET_USERNAME    Username for credentialed PhysioNet downloads.
  MIMIC_CXR_FILTER      Optional grep filter for IMAGE_FILENAMES to pull only a subset.
  ALLOW_KAGGLE_MIRRORS  If set to 1, use Kaggle mirrors when the official source is not automation-friendly.

Examples:
  ${SCRIPT_NAME} all
  ${SCRIPT_NAME} shenzhen rsna-pneumonia
  PHYSIONET_USERNAME=<user> ${SCRIPT_NAME} mimic-cxr-jpg
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

download_mimic_cxr_jpg() {
  local out="${DOMAIN_ROOT}/MIMIC-CXR-JPG"
  if dataset_done_with_dir "$out"; then
    log "Skipping MIMIC-CXR-JPG: marker and payload already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"

  if have_cmd kaggle && have_kaggle_auth; then
    log "Downloading MIMIC-CXR-JPG via Kaggle mirror to ${out}"
    download_kaggle_dataset "simhadrisadaram/mimic-cxr-dataset" "$out"
    mark_dataset_done "$out"
    return 0
  fi

  if ! download_url "https://physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES" "${out}/IMAGE_FILENAMES"; then
    warn "Skipping MIMIC-CXR-JPG: Kaggle auth unavailable and PhysioNet IMAGE_FILENAMES could not be fetched from this host"
    return 0
  fi
  if ! have_cmd wget || [[ -z "${PHYSIONET_USERNAME:-}" ]]; then
    warn "Skipping MIMIC-CXR-JPG: Kaggle auth unavailable and PHYSIONET_USERNAME/wget are not configured"
    return 0
  fi

  local selected_file="${out}/IMAGE_FILENAMES"
  if [[ -n "${MIMIC_CXR_FILTER:-}" ]]; then
    grep "${MIMIC_CXR_FILTER}" "${out}/IMAGE_FILENAMES" > "${out}/SELECTED_FILES" || true
    if [[ ! -s "${out}/SELECTED_FILES" ]]; then
      warn "MIMIC_CXR_FILTER matched no records; leaving note file instead"
      write_manual_note "$out" \
        "Official source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/" \
        "Your MIMIC_CXR_FILTER=${MIMIC_CXR_FILTER} matched no IMAGE_FILENAMES entries." \
        "Inspect ${out}/IMAGE_FILENAMES and rerun with a narrower or corrected filter."
      return 0
    fi
    selected_file="${out}/SELECTED_FILES"
  fi

  log "Downloading MIMIC-CXR-JPG into ${out}"
  (
    cd "$out"
    wget -r -N -c -np -nH --cut-dirs=1 \
      --user "${PHYSIONET_USERNAME}" --ask-password \
      -i "$(basename "${selected_file}")" \
      --base="https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
  )
  mark_dataset_done "$out"
}

download_chexpert() {
  local out="${DOMAIN_ROOT}/CheXpert"
  if dataset_done_with_dir "$out"; then
    log "Skipping CheXpert: marker and payload already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"
  if have_cmd kaggle && have_kaggle_auth; then
    log "Downloading CheXpert via Kaggle mirror to ${out}"
    download_kaggle_dataset "ashery/chexpert" "$out"
    mark_dataset_done "$out"
    return 0
  fi
  if have_cmd hf; then
    log "Downloading CheXpert via Hugging Face mirror to ${out}"
    download_hf_dataset "danjacobellis/chexpert" "$out"
    mark_dataset_done "$out"
    return 0
  fi
  warn "Skipping CheXpert: neither Kaggle auth nor hf CLI is available"
}

download_chestxray14() {
  local out="${DOMAIN_ROOT}/ChestXray14"
  if dataset_done_with_dir "$out"; then
    log "Skipping ChestXray14: marker and payload already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"

  if have_cmd hf; then
    log "Downloading ChestXray14 via Hugging Face mirror to ${out}"
    download_hf_dataset "alkzar90/NIH-Chest-X-ray-dataset" "$out"
    mark_dataset_done "$out"
    return 0
  fi

  if [[ "${ALLOW_KAGGLE_MIRRORS:-0}" == "1" ]] && have_cmd kaggle && have_kaggle_auth; then
    log "Downloading ChestXray14 via Kaggle mirror to ${out}"
    if ! kaggle datasets download nih-chest-xrays/data -p "$out" --unzip; then
      kaggle datasets download -d nih-chest-xrays/data -p "$out" --unzip
    fi
    mark_dataset_done "$out"
    return 0
  fi

  warn "Skipping ChestXray14: no active Hugging Face or Kaggle mirror is available"
}

download_montgomery() {
  local out="${DOMAIN_ROOT}/Montgomery"
  local base_url="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/"
  if [[ -f "${out}/.done" ]] && montgomery_complete "$out"; then
    log "Skipping Montgomery: marker and extracted payload already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"
  rm -f "${out}/NLM-MontgomeryCXRSet.zip"
  log "Downloading Montgomery County CXR set to ${out}"
  download_montgomery_listing "${base_url}index.html" "${base_url}" "${out}"
  download_montgomery_listing "${base_url}CXR_png/index.html" "${base_url}CXR_png/" "${out}/CXR_png"
  download_montgomery_listing "${base_url}ClinicalReadings/index.html" "${base_url}ClinicalReadings/" "${out}/ClinicalReadings"
  download_montgomery_listing "${base_url}ManualMask/index.html" "${base_url}ManualMask/" "${out}/ManualMask"
  download_montgomery_listing "${base_url}ManualMask/leftMask/index.html" "${base_url}ManualMask/leftMask/" "${out}/ManualMask/leftMask"
  download_montgomery_listing "${base_url}ManualMask/rightMask/index.html" "${base_url}ManualMask/rightMask/" "${out}/ManualMask/rightMask"
  montgomery_complete "$out" || die "Montgomery download appears incomplete; not writing .done"
  mark_dataset_done "$out"
}

download_montgomery_listing() {
  local index_url="$1"
  local base_url="$2"
  local out_dir="$3"
  local tmp
  tmp="$(mktemp)"
  mkdir -p "$out_dir"
  download_url "$index_url" "$tmp"
  mapfile -t rel_paths < <(grep -o "href='[^']*'" "$tmp" | sed "s/^href='//; s/'$//" | grep -vE '(^|/)index\\.html$' || true)
  rm -f "$tmp"
  local rel
  for rel in "${rel_paths[@]}"; do
    download_url "${base_url}${rel}" "${out_dir}/${rel}"
  done
}

montgomery_complete() {
  local out="$1"
  [[ -d "${out}/CXR_png" && -d "${out}/ClinicalReadings" && -d "${out}/ManualMask" ]] || return 1
  [[ -f "${out}/montgomery_consensus_roi.csv" ]] || return 1
  local n_img n_txt n_mask
  n_img="$(find "${out}/CXR_png" -type f | wc -l | tr -d ' ')"
  n_txt="$(find "${out}/ClinicalReadings" -type f | wc -l | tr -d ' ')"
  n_mask="$(find "${out}/ManualMask" -type f | wc -l | tr -d ' ')"
  [[ "${n_img}" -ge 100 && "${n_txt}" -ge 100 && "${n_mask}" -ge 200 ]]
}

download_shenzhen() {
  local out="${DOMAIN_ROOT}/Shenzhen"
  local archive="${out}/ChinaSet_AllFiles.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping Shenzhen: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading Shenzhen Hospital CXR set to ${out}"
  download_url "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip" "$archive"
  mark_dataset_done "$out"
}

download_brixia() {
  local out="${DOMAIN_ROOT}/BrixIA-COVID19"
  write_manual_note "$out" \
    "Official source: https://physionet.org/content/brixia-covid19/1.0.0/" \
    "Use the PhysioNet page for the current access flow and download instructions."
}

download_bimcv() {
  local out="${DOMAIN_ROOT}/BIMCV-COVID19+"
  write_manual_note "$out" \
    "Official source: https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/" \
    "Use the official BIMCV portal for the current download / registration flow."
}

download_tb_portals() {
  local out="${DOMAIN_ROOT}/TB-Portals"
  write_manual_note "$out" \
    "Official source: https://tbportals.niaid.nih.gov/download-data" \
    "TB Portals provides download guidance and current package links on the official portal."
}

download_rsna_pneumonia() {
  local out="${DOMAIN_ROOT}/RSNA-Pneumonia"
  local archive="${out}/rsna-pneumonia-detection-challenge.zip"
  if dataset_done_with_dir "$out"; then
    log "Skipping RSNA Pneumonia: marker and payload already present"
    return 0
  fi
  mkdir -p "$out"
  clear_manual_note "$out"
  if have_cmd kaggle && have_kaggle_auth; then
    log "Downloading RSNA Pneumonia via Kaggle dataset mirror to ${out}"
    download_kaggle_dataset "aldezo313/rsna-pneumonia-dataset" "$out"
    mark_dataset_done "$out"
    return 0
  fi
  if have_cmd hf; then
    log "Downloading RSNA Pneumonia via Hugging Face mirror to ${out}"
    download_hf_dataset "Baldezo313/rsna-pneumonia-dataset" "$out"
    mark_dataset_done "$out"
    return 0
  fi
  warn "Skipping RSNA Pneumonia: neither Kaggle auth nor hf CLI is available"
}

download_midrc() {
  local out="${DOMAIN_ROOT}/MIDRC"
  write_manual_note "$out" \
    "Official source: https://midrc.org/" \
    "MIDRC access and export are portal-driven and may require approval depending on the collection."
}

download_fairfedmed_chest() {
  local out="${DOMAIN_ROOT}/FairFedMed-Chest"
  write_manual_note "$out" \
    "Official project page: https://github.com/Harvard-AI-and-Robotics-Lab/FairFedMed" \
    "The paper describes FairFedMed-Chest as a constructed subset over CheXpert and MIMIC-CXR rather than a simple standalone public zip." \
    "Rebuild or request it from the official project resources."
}

expand_targets() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      all)
        printf '%s\n' mimic-cxr-jpg chexpert chestxray14 montgomery shenzhen brixia-covid19 bimcv-covid19 tb-portals rsna-pneumonia midrc fairfedmed-chest
        ;;
      mimic-cxr-jpg|chexpert|chestxray14|montgomery|shenzhen|brixia-covid19|bimcv-covid19|tb-portals|rsna-pneumonia|midrc|fairfedmed-chest)
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
    mimic-cxr-jpg) download_mimic_cxr_jpg ;;
    chexpert) download_chexpert ;;
    chestxray14) download_chestxray14 ;;
    montgomery) download_montgomery ;;
    shenzhen) download_shenzhen ;;
    brixia-covid19) download_brixia ;;
    bimcv-covid19) download_bimcv ;;
    tb-portals) download_tb_portals ;;
    rsna-pneumonia) download_rsna_pneumonia ;;
    midrc) download_midrc ;;
    fairfedmed-chest) download_fairfedmed_chest ;;
  esac
done

log "Done."
