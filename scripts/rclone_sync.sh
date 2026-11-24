#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Penggunaan: scripts/rclone_sync.sh upload|download|both [--execute]

  upload    Akar repo lokal -> residual:
  download  residual: -> akar repo lokal
  both      Download, lalu upload

Tanpa --execute, semua langkah memakai --dry-run. Tidak ada penghapusan.
File tujuan hanya ditimpa bila file sumber lebih baru (--update).
Audio, cache .npz, dan arsip tidak pernah ditransfer (lihat configs/rclone-sync.filter).
Nama remote dapat diganti dengan RCLONE_REMOTE (default: residual:).
USAGE
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
mode=$1
case "$mode" in
  upload|download|both) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
execute=false
if [[ $# -eq 2 ]]; then
  [[ $2 == --execute ]] || { usage >&2; exit 2; }
  execute=true
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
filter="$root/configs/rclone-sync.filter"
remote=${RCLONE_REMOTE:-residual:}
[[ -f $filter ]] || { echo "Filter tidak ditemukan: $filter" >&2; exit 1; }
command -v rclone >/dev/null || { echo 'rclone tidak ditemukan di PATH.' >&2; exit 1; }

copy_one_way() {
  local args=(copy "$1" "$2" --filter-from "$filter" --ignore-case --create-empty-src-dirs --update)
  if [[ $execute == true ]]; then args+=(--progress); else args+=(--dry-run -v); fi
  printf '\n%s -> %s (%s)\n' "$1" "$2" "$([[ $execute == true ]] && echo transfer || echo simulasi)"
  rclone "${args[@]}"
}

case "$mode" in
  upload) copy_one_way "$root" "$remote" ;;
  download) copy_one_way "$remote" "$root" ;;
  both) copy_one_way "$remote" "$root"; copy_one_way "$root" "$remote" ;;
esac
