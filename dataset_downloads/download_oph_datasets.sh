#!/usr/bin/env bash
set -euo pipefail

# Download script for publicly reachable ophthalmology datasets mentioned in
# [NPJ-DM投稿] Fairness in Ophthalmology Foundation Models.pdf.
#
# Included here:
# - EyePACS (Kaggle competition)
# - APTOS 2019 (Kaggle competition)
# - ODIR-5K Kaggle mirror
# - Harvard-GF (Hugging Face dataset)
# - FairFedMed-Oph (Hugging Face dataset)
# - IDRiD (Hugging Face dataset)
# - GRAPE (Figshare files)
# - RFMiD / RIADD (Google Drive links from challenge page)
# - JSIEC (Zenodo direct zip)
# - OculoScope (Figshare zip from FairerOPTH release)
# - MixNAF (Figshare zip from FairerOPTH release)
#
# Not included here because they are not direct CLI downloads:
# - AREDS (dbGaP authorized access)
# - UK Biobank (application-based access)
# - OHTS (request/approval based)
# - Grand Challenge official ODIR / iChallenge-AMD direct endpoints
#
# Prerequisites:
# - Kaggle datasets/competitions: `pip install kaggle`
#   Also configure Kaggle auth and accept competition rules in the browser.
# - Hugging Face datasets: `pip install "huggingface_hub[cli]"`
# - RFMiD Google Drive links: `pip install gdown`
# - GRAPE: `curl` or `wget`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
source "${SCRIPT_DIR}/download_dataset_common.sh"

DOMAIN_SLUG="ophthalmology"
DOMAIN_ROOT="$(resolve_domain_root "${DOMAIN_SLUG}")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [DATASET ...]

Domain root:
  ${DOMAIN_ROOT}

Datasets:
  eyepacs
  aptos2019
  odir5k
  harvard-gf
  fairfedmed-oph
  grape
  rfmid
  idrid
  jsiec
  oculoscope
  mixnaf
  all

Examples:
  ${SCRIPT_NAME} all
  ${SCRIPT_NAME} harvard-gf fairfedmed-oph
  ${SCRIPT_NAME} idrid
  ${SCRIPT_NAME} jsiec oculoscope mixnaf
  DOWNLOAD_ROOT=/data/datasets ${SCRIPT_NAME} eyepacs aptos2019

Environment:
  DOWNLOAD_ROOT   Base root directory. Final path becomes \$DOWNLOAD_ROOT/${DOMAIN_SLUG}
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

download_eyepacs() {
  local out="${DOMAIN_ROOT}/EyePACS"
  local archive="${out}/diabetic-retinopathy-detection.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping EyePACS: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading EyePACS competition bundle to ${out}"
  log "If this fails, accept competition rules first: https://www.kaggle.com/competitions/diabetic-retinopathy-detection"
  download_kaggle_competition_file "diabetic-retinopathy-detection" "$archive"
  mark_dataset_done "$out"
}

download_aptos2019() {
  local out="${DOMAIN_ROOT}/APTOS2019"
  local archive="${out}/aptos2019-blindness-detection.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping APTOS2019: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading APTOS 2019 competition bundle to ${out}"
  log "If this fails, accept competition rules first: https://www.kaggle.com/competitions/aptos2019-blindness-detection"
  download_kaggle_competition_file "aptos2019-blindness-detection" "$archive"
  mark_dataset_done "$out"
}

download_odir5k() {
  local out="${DOMAIN_ROOT}/ODIR5K"
  if dataset_done_with_dir "$out"; then
    log "Skipping ODIR-5K: marker and extracted contents already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading ODIR-5K Kaggle mirror to ${out}"
  if ! kaggle datasets download andrewmvd/ocular-disease-recognition-odir5k -p "$out" --unzip; then
    kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k -p "$out" --unzip
  fi
  mark_dataset_done "$out"
}

download_harvard_gf() {
  local out="${DOMAIN_ROOT}/Harvard-GF"
  local archive="${out}/Dataset/dataset.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping Harvard-GF: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading Harvard-GF to ${out}"
  download_hf_file "harvardairobotics/Harvard-GF" "Dataset/dataset.zip" "$out"
  mark_dataset_done "$out"
}

download_fairfedmed_oph() {
  local out="${DOMAIN_ROOT}/FairFedMed-Oph"
  local archive="${out}/FairFedMed-Oph/Dataset/dataset.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping FairFedMed-Oph: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading FairFedMed-Oph to ${out}"
  download_hf_file "harvardairobotics/FairFedMed" "FairFedMed-Oph/Dataset/dataset.zip" "$out"
  mark_dataset_done "$out"
}

download_grape() {
  local out="${DOMAIN_ROOT}/GRAPE"
  local f1="${out}/VF_and_clinical_information.xlsx"
  local f2="${out}/CFPs.rar"
  local f3="${out}/ROI_images.rar"
  local f4="${out}/Annotated_Images.rar"
  local f5="${out}/json.rar"
  if dataset_done_with_files "$out" "$f1" "$f2" "$f3" "$f4" "$f5"; then
    log "Skipping GRAPE: marker and files already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading GRAPE files to ${out}"
  download_url "https://ndownloader.figshare.com/files/41670009" "$f1"
  download_url "https://ndownloader.figshare.com/files/41358156" "$f2"
  download_url "https://ndownloader.figshare.com/files/41358150" "$f3"
  download_url "https://ndownloader.figshare.com/files/41358159" "$f4"
  download_url "https://ndownloader.figshare.com/files/41358162" "$f5"
  mark_dataset_done "$out"
}

download_rfmid() {
  local out="${DOMAIN_ROOT}/RFMiD"
  local train_zip="${out}/train_all_46_classes.zip"
  local eval_zip="${out}/eval_all_46_classes.zip"
  local test_zip="${out}/test_all_46_classes.zip"
  if dataset_done_with_files "$out" "$train_zip" "$eval_zip" "$test_zip"; then
    log "Skipping RFMiD: marker and files already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading RFMiD bundles to ${out}"
  download_gdown_file "https://drive.google.com/file/d/1_pzNv3z3H1_ao8nOrp-B40S5TE1Rcx1W/view?usp=sharing" "$train_zip"
  download_gdown_file "https://drive.google.com/file/d/1ORmTGxwXrU9QjufVL8H93yj8GZHjxrXC/view?usp=sharing" "$eval_zip"
  download_gdown_file "https://drive.google.com/file/d/167Km-NVmvQLlbegWKC3Biwu0ETs4dWUK/view?usp=sharing" "$test_zip"
  mark_dataset_done "$out"
}

download_idrid() {
  local out="${DOMAIN_ROOT}/IDRiD"
  local seg_zip="${out}/A.Segmentation.zip"
  local grading_zip="${out}/B.Disease_Grading.zip"
  local loc_zip="${out}/C.Localization.zip"
  if dataset_done_with_files "$out" "$seg_zip" "$grading_zip" "$loc_zip"; then
    log "Skipping IDRiD: marker and files already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading IDRiD from Hugging Face to ${out}"
  download_hf_file "MahsaTorki/IDRiD_Dataset" "A.Segmentation.zip" "$out"
  download_hf_file "MahsaTorki/IDRiD_Dataset" "B.Disease_Grading.zip" "$out"
  download_hf_file "MahsaTorki/IDRiD_Dataset" "C.Localization.zip" "$out"
  mark_dataset_done "$out"
}

download_jsiec() {
  local out="${DOMAIN_ROOT}/JSIEC"
  local archive="${out}/1000images.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping JSIEC: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading JSIEC zip to ${out}"
  download_url "https://zenodo.org/records/3477553/files/1000images.zip?download=1" "$archive"
  mark_dataset_done "$out"
}

download_oculoscope() {
  local out="${DOMAIN_ROOT}/OculoScope"
  local archive="${out}/OculoScope.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping OculoScope: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading OculoScope release zip to ${out}"
  download_url "https://ndownloader.figshare.com/files/43334379" "$archive"
  mark_dataset_done "$out"
}

download_mixnaf() {
  local out="${DOMAIN_ROOT}/MixNAF"
  local archive="${out}/MixNAF.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping MixNAF: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading MixNAF release zip to ${out}"
  download_url "https://ndownloader.figshare.com/files/43426356" "$archive"
  mark_dataset_done "$out"
}

expand_targets() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      all)
        printf '%s\n' eyepacs aptos2019 odir5k harvard-gf fairfedmed-oph grape rfmid idrid jsiec oculoscope mixnaf
        ;;
      eyepacs|aptos2019|odir5k|harvard-gf|fairfedmed-oph|grape|rfmid|idrid|jsiec|oculoscope|mixnaf)
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
    eyepacs) download_eyepacs ;;
    aptos2019) download_aptos2019 ;;
    odir5k) download_odir5k ;;
    harvard-gf) download_harvard_gf ;;
    fairfedmed-oph) download_fairfedmed_oph ;;
    grape) download_grape ;;
    rfmid) download_rfmid ;;
    idrid) download_idrid ;;
    jsiec) download_jsiec ;;
    oculoscope) download_oculoscope ;;
    mixnaf) download_mixnaf ;;
  esac
done

log "Done."
