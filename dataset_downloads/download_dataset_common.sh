#!/usr/bin/env bash
# shellcheck shell=bash

set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

default_download_root() {
  printf '%s\n' "${PWD}/datasets"
}

log() {
  printf '[%s] %s\n' "${SCRIPT_NAME:-download}" "$*"
}

warn() {
  printf '[%s] WARNING: %s\n' "${SCRIPT_NAME:-download}" "$*" >&2
}

die() {
  printf '[%s] ERROR: %s\n' "${SCRIPT_NAME:-download}" "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

need_cmd() {
  have_cmd "$1" || die "Missing command: $1"
}

need_http_client() {
  if have_cmd curl; then
    HTTP_CLIENT="curl"
  elif have_cmd wget; then
    HTTP_CLIENT="wget"
  else
    die "Need either curl or wget"
  fi
}

need_kaggle() {
  need_cmd kaggle
  if [[ -z "${KAGGLE_API_TOKEN:-}" && -z "${KAGGLE_USERNAME:-}" && -z "${KAGGLE_KEY:-}" \
        && ! -f "${HOME}/.kaggle/access_token" && ! -f "${HOME}/.kaggle/kaggle.json" \
        && ! -f "${HOME}/.config/kaggle/kaggle.json" ]]; then
    die "Kaggle auth not found. Configure ~/.kaggle/access_token or ~/.kaggle/kaggle.json first."
  fi
}

have_kaggle_auth() {
  [[ -n "${KAGGLE_API_TOKEN:-}" || ( -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ) \
     || -f "${HOME}/.kaggle/access_token" || -f "${HOME}/.kaggle/kaggle.json" \
     || -f "${HOME}/.config/kaggle/kaggle.json" ]]
}

need_hf() {
  need_cmd hf
}

need_gdown() {
  need_cmd gdown
}

is_nonempty_file() {
  local path="$1"
  [[ -f "$path" && -s "$path" ]]
}

dir_has_payload() {
  local path="$1"
  [[ -d "$path" ]] && find "$path" -mindepth 1 ! -name '.done' ! -name 'ACCESS_INSTRUCTIONS.txt' -print -quit | grep -q .
}

dataset_done_with_files() {
  local out="$1"
  shift
  [[ -f "${out}/.done" ]] || return 1
  local path
  for path in "$@"; do
    is_nonempty_file "$path" || return 1
  done
  return 0
}

dataset_done_with_dir() {
  local out="$1"
  [[ -f "${out}/.done" ]] || return 1
  dir_has_payload "$out"
}

mark_dataset_done() {
  local out="$1"
  mkdir -p "$out"
  touch "${out}/.done"
}

download_url() {
  local url="$1"
  local output="$2"
  need_http_client
  if is_nonempty_file "$output"; then
    log "Skipping existing file ${output}"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  if [[ "${HTTP_CLIENT}" == "curl" ]]; then
    curl -L --fail --retry 3 --retry-delay 2 "$url" -o "$output"
  else
    wget -O "$output" "$url"
  fi
}

download_kaggle_competition_file() {
  local competition="$1"
  local output="$2"
  if is_nonempty_file "$output"; then
    log "Skipping existing file ${output}"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  kaggle competitions download -c "$competition" -p "$(dirname "$output")"
}

download_kaggle_dataset() {
  local dataset="$1"
  local out="$2"
  mkdir -p "$out"
  kaggle datasets download "$dataset" -p "$out" --unzip
}

download_hf_dataset() {
  local dataset="$1"
  local out="$2"
  shift 2
  need_hf
  mkdir -p "$out"
  local cmd=(hf download "$dataset" --repo-type dataset --local-dir "$out")
  local pattern
  for pattern in "$@"; do
    cmd+=(--include "$pattern")
  done
  "${cmd[@]}"
}

download_hf_file() {
  local repo="$1"
  local filename="$2"
  local out="$3"
  local output="${out}/${filename}"
  need_hf
  if is_nonempty_file "$output"; then
    log "Skipping existing file ${output}"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  hf download "$repo" "$filename" --repo-type dataset --local-dir "$out"
}

download_gdown_file() {
  local url="$1"
  local output="$2"
  need_gdown
  if is_nonempty_file "$output"; then
    log "Skipping existing file ${output}"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  gdown --fuzzy "$url" -O "$output"
}

download_dataverse_file() {
  local file_id="$1"
  local output="$2"
  download_url "https://dataverse.harvard.edu/api/access/datafile/${file_id}" "$output"
}

write_manual_note() {
  local out="$1"
  shift
  mkdir -p "$out"
  {
    printf 'This dataset is not wired for direct unattended download in this script.\n'
    printf 'Reason: the official source requires manual approval, interactive login, or per-user portal setup.\n\n'
    printf 'Current guidance:\n'
    local line
    for line in "$@"; do
      printf -- '- %s\n' "$line"
    done
  } > "${out}/ACCESS_INSTRUCTIONS.txt"
  log "Wrote manual access instructions to ${out}/ACCESS_INSTRUCTIONS.txt"
}

clear_manual_note() {
  local out="$1"
  rm -f "${out}/ACCESS_INSTRUCTIONS.txt"
}

resolve_domain_root() {
  local domain_slug="$1"
  local base_root="${DOWNLOAD_ROOT:-$(default_download_root)}"
  printf '%s/%s\n' "${base_root}" "${domain_slug}"
}
