#!/usr/bin/env bash
# Download human HYPOMAP snRNA-seq data (Cambridge Apollo Repository)
# Source: https://www.repository.cam.ac.uk/items/cad1c61a-e4e5-4443-ad11-92e4f48b3861
# File: human_HYPOMAP_snRNASeq.h5ad (~6.18 GB)

set -euo pipefail

OUT_DIR="data/human_hypomap"
OUT_FILE="${OUT_DIR}/human_HYPOMAP_snRNASeq.h5ad"
URL="https://www.repository.cam.ac.uk/bitstreams/07f66aef-76cd-48aa-b0ae-9a7a201dbbc0/download"
EXPECTED_BYTES=6639007892

mkdir -p "${OUT_DIR}"

# -C - resumes a partial download if the connection drops
curl -L -C - --retry 10 --retry-delay 5 -o "${OUT_FILE}" "${URL}"

actual_bytes=$(stat -f%z "${OUT_FILE}" 2>/dev/null || stat -c%s "${OUT_FILE}")
if [ "${actual_bytes}" != "${EXPECTED_BYTES}" ]; then
    echo "WARNING: size mismatch — got ${actual_bytes} bytes, expected ${EXPECTED_BYTES}" >&2
    exit 1
fi

file "${OUT_FILE}"
echo "Downloaded and verified: ${OUT_FILE} (${actual_bytes} bytes)"
