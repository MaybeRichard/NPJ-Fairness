#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
source "${SCRIPT_DIR}/download_dataset_common.sh"

DOMAIN_SLUG="general"
DOMAIN_ROOT="$(resolve_domain_root "${DOMAIN_SLUG}")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [DATASET ...]

Domain root:
  ${DOMAIN_ROOT}

Datasets:
  brfss2021
  adult-income
  cifar10
  cifar100
  stanford-dogs
  celeba
  all

Examples:
  ${SCRIPT_NAME} all
  ${SCRIPT_NAME} brfss2021 adult-income
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

download_brfss2021() {
  local out="${DOMAIN_ROOT}/BRFSS2021"
  local archive="${out}/LLCP2021XPT.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping BRFSS2021: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading BRFSS 2021 to ${out}"
  download_url "https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip" "$archive"
  mark_dataset_done "$out"
}

download_adult_income() {
  local out="${DOMAIN_ROOT}/Adult-Income"
  local archive="${out}/adult.zip"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping Adult Income: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading UCI Adult dataset to ${out}"
  download_url "https://archive.ics.uci.edu/static/public/2/adult.zip" "$archive"
  mark_dataset_done "$out"
}

download_cifar10() {
  local out="${DOMAIN_ROOT}/CIFAR-10"
  local archive="${out}/cifar-10-python.tar.gz"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping CIFAR-10: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading CIFAR-10 to ${out}"
  download_url "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" "$archive"
  mark_dataset_done "$out"
}

download_cifar100() {
  local out="${DOMAIN_ROOT}/CIFAR-100"
  local archive="${out}/cifar-100-python.tar.gz"
  if dataset_done_with_files "$out" "$archive"; then
    log "Skipping CIFAR-100: marker and archive already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading CIFAR-100 to ${out}"
  download_url "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" "$archive"
  mark_dataset_done "$out"
}

download_stanford_dogs() {
  local out="${DOMAIN_ROOT}/Stanford-Dogs"
  local images="${out}/images.tar"
  local lists="${out}/lists.tar"
  if dataset_done_with_files "$out" "$images" "$lists"; then
    log "Skipping Stanford Dogs: marker and archives already present"
    return 0
  fi
  mkdir -p "$out"
  log "Downloading Stanford Dogs to ${out}"
  download_url "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar" "$images"
  download_url "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar" "$lists"
  mark_dataset_done "$out"
}

download_celeba() {
  local out="${DOMAIN_ROOT}/CelebA"
  write_manual_note "$out" \
    "Official source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" \
    "CelebA is distributed through the official project page / Google Drive links and is not wired for unattended download here."
}

expand_targets() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      all)
        printf '%s\n' brfss2021 adult-income cifar10 cifar100 stanford-dogs celeba
        ;;
      brfss2021|adult-income|cifar10|cifar100|stanford-dogs|celeba)
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
    brfss2021) download_brfss2021 ;;
    adult-income) download_adult_income ;;
    cifar10) download_cifar10 ;;
    cifar100) download_cifar100 ;;
    stanford-dogs) download_stanford_dogs ;;
    celeba) download_celeba ;;
  esac
done

log "Done."
