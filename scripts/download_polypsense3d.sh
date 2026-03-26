#!/usr/bin/env bash
# Download PolypSense3D subsets from Harvard Dataverse.
#
# Source : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K13H89
# License: CC0 1.0 (Public Domain)
# Paper  : Zhang et al., "PolypSense3D: A Multi-Source Benchmark Dataset for
#           Depth-Aware Polyp Size Measurement in Endoscopy", NeurIPS 2025
#
# Subsets:
#   virtual  — synthetic (Stage 1 training, has GT depth)
#   clinical — real clinical colonoscopy (Stage 2b, no GT depth)
#   physical — physical phantom (59 MB)
#
# Usage:
#   bash scripts/download_polypsense3d.sh [--dest /path/to/data/PolypSense3D] [--subset virtual|clinical|physical|all]
#
# Requirements:
#   curl
#   p7zip (7za/7zz) OR python3 with py7zr  (pip install py7zr)  — for .7z files
#   unrar  — for .rar files (sudo apt install unrar)

set -euo pipefail

BASE_URL="https://dataverse.harvard.edu"
DEST="/home/in4218/code/data/PolypSense3D"
SUBSET="virtual"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)   DEST="$2";   shift 2 ;;
        --subset) SUBSET="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "$DEST"

# ── file registry (IDs from doi:10.7910/DVN/K13H89 v5) ───────────────────────
declare -A FILE_IDS=(
    [virtual]="11383671"
    [physical]="11383623"
    [clinical]="12556995"
)
declare -A FILE_NAMES=(
    [virtual]="Virtual-Dataset-For-PolypSense3D.rar"
    [physical]="Physical-Dataset-For-PolypSense3D.rar"
    [clinical]="Clinical-Dataset-For-PolypSense3D.7z"
)
declare -A FILE_SIZES=(
    [virtual]="~1.1 GB"
    [physical]="~59 MB"
    [clinical]="~2.5 GB"
)

if [[ "$SUBSET" == "all" ]]; then
    SUBSETS=("virtual" "physical" "clinical")
else
    SUBSETS=("$SUBSET")
fi

# ── extract helper ────────────────────────────────────────────────────────────
extract() {
    local archive="$1"
    local dest="$2"
    local ext="${archive##*.}"

    if [[ "$ext" == "7z" ]]; then
        if command -v 7za &>/dev/null; then
            7za x "$archive" -o"$dest" -y
        elif command -v 7zz &>/dev/null; then
            7zz x "$archive" -o"$dest" -y
        else
            echo "    (7za/7zz not found — using py7zr)"
            python3 - <<PYEOF
import sys
try:
    import py7zr
except ImportError:
    print("Error: py7zr not installed. Run:  pip install py7zr", file=sys.stderr)
    sys.exit(1)
with py7zr.SevenZipFile("$archive", mode="r") as z:
    z.extractall(path="$dest")
PYEOF
        fi
    elif [[ "$ext" == "rar" ]]; then
        if command -v unrar &>/dev/null; then
            unrar x "$archive" "$dest/"
        else
            echo "Error: unrar not installed. Run:  sudo apt install unrar" >&2
            exit 1
        fi
    else
        echo "Error: unknown archive format: $archive" >&2
        exit 1
    fi
}

# ── download + extract each subset ───────────────────────────────────────────
for subset in "${SUBSETS[@]}"; do
    FILE_ID="${FILE_IDS[$subset]}"
    FILE_NAME="${FILE_NAMES[$subset]}"
    FILE_SIZE="${FILE_SIZES[$subset]}"
    ARCHIVE="$DEST/$FILE_NAME"

    echo ""
    echo "==> [$subset] Downloading $FILE_NAME ($FILE_SIZE) …"
    curl -L --progress-bar \
        "$BASE_URL/api/access/datafile/$FILE_ID" \
        -o "$ARCHIVE"

    echo "==> [$subset] Extracting …"
    extract "$ARCHIVE" "$DEST"
    rm "$ARCHIVE"
    echo "==> [$subset] Done → $DEST"
done

echo ""
echo "All requested subsets downloaded to: $DEST"
echo ""
echo "Quick check (virtual subset):"
echo "  python -c \""
echo "    from endo_da3.data.polypsense3d import PolypSense3DVirtualDataset"
echo "    ds = PolypSense3DVirtualDataset('$DEST', split='train')"
echo "    print(len(ds), ds[0]['images'].shape)"
echo "  \""
