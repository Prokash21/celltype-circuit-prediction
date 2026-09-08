#!/usr/bin/env bash
# Download mouse HypoMap snRNA-seq data (CZ CELLxGENE Discover)
# Collection: https://cellxgene.cziscience.com/collections/d86517f0-fa7e-4266-b82e-a521350d6d36
# Dataset: HypoMap - unified single cell atlas of the murine hypothalamus (384,925 cells)
#
# Note: the Cambridge repository (https://www.repository.cam.ac.uk/handle/1810/340518)
# only hosts this dataset as a Seurat .rds object. CZ CELLxGENE provides the same
# atlas as an h5ad/AnnData export, which is used here.

set -euo pipefail

OUT_DIR="data/mouse_hypomap"
OUT_FILE="${OUT_DIR}/mouse_HypoMap_snRNASeq.h5ad"
URL="https://datasets.cellxgene.cziscience.com/87b802cc-73ca-422a-8cc7-6d6d38449b3f.h5ad"
EXPECTED_BYTES=3814401378

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
