#!/usr/bin/env bash
# Sanity-check both hypothalamus datasets after download:
# file size matches the source, and the file is a valid HDF5 container.

set -euo pipefail

check() {
    local path="$1"
    local expected_bytes="$2"

    if [ ! -f "${path}" ]; then
        echo "MISSING: ${path}" >&2
        return 1
    fi

    local actual_bytes
    actual_bytes=$(stat -f%z "${path}" 2>/dev/null || stat -c%s "${path}")

    if [ "${actual_bytes}" != "${expected_bytes}" ]; then
        echo "SIZE MISMATCH: ${path} — got ${actual_bytes}, expected ${expected_bytes}" >&2
        return 1
    fi

    if ! file "${path}" | grep -q "Hierarchical Data Format"; then
        echo "NOT VALID HDF5: ${path}" >&2
        return 1
    fi

    echo "OK: ${path} (${actual_bytes} bytes, valid HDF5)"
}

check "data/human_hypomap/human_HYPOMAP_snRNASeq.h5ad" 6639007892
check "data/mouse_hypomap/mouse_HypoMap_snRNASeq.h5ad" 3814401378
