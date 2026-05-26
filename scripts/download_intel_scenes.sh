#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
ZIP_PATH="${DATA_DIR}/intel-image-classification.zip"
URL="${INTEL_SCENES_URL:-https://hf-mirror.com/datasets/resolverkatla/Intel-Image-Classification/resolve/main/intel-image-classification.zip}"

mkdir -p "${DATA_DIR}"

if command -v wget >/dev/null 2>&1; then
  wget -c --timeout=60 --tries=20 --waitretry=5 -O "${ZIP_PATH}" "${URL}"
else
  curl -L --fail --retry 20 --retry-delay 5 --connect-timeout 60 -C - -o "${ZIP_PATH}" "${URL}"
fi

unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}/intel-scenes"

echo "Dataset extracted to ${DATA_DIR}/intel-scenes"
find "${DATA_DIR}/intel-scenes" -maxdepth 3 -type d | sort
