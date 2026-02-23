#!/usr/bin/env bash
# Download the PolypSense3D Virtual dataset from Harvard Dataverse.
#
# Source : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LKDIEK
# License: CC0 1.0 (Public Domain)
# Paper  : Zhang et al., "PolypSense3D: A Multi-Source Benchmark Dataset for
#           Depth-Aware Polyp Size Measurement in Endoscopy", NeurIPS 2025
#
# Only the Virtual subset is downloaded — the other two subsets (Clinical,
# Physical) do not contain the depth_estimation layout used in Stage 1.
#
# Usage:
#   bash scripts/download_polypsense3d.sh [--dest /path/to/data/PolypSense3D]
#
# Result layout:
#   <dest>/Virtual Dataset For PolypSense3D/depth_estimation/
#       camera.txt                  9 space-separated floats (3×3 K, row-major)
#       position_rotation.csv       (tX,tY,tZ,rX,rY,rZ,rW,time)
#       images/image_NNNN.jpg       320×320 RGB  (~8 241 frames)
#       depths/aov_image_NNNN.png   320×320 RGBA uint8 (R = depth in mm)
#
# Requirements:
#   curl
#   p7zip (7za) OR python3 with py7zr  (pip install py7zr)

set -euo pipefail

BASE_URL="https://dataverse.harvard.edu"
PERSISTENT_ID="doi:10.7910/DVN/LKDIEK"
VIRTUAL_FILE="Virtual-Dataset-For-PolypSense3D.7z"
DEST="${1:-/home/in4218/code/data/PolypSense3D}"

# Override dest with --dest flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        *) shift ;;
    esac
done

DATA_DIR="$DEST/Virtual Dataset For PolypSense3D/depth_estimation"

# ── idempotency check ─────────────────────────────────────────────────────────
if [[ -d "$DATA_DIR/images" && -d "$DATA_DIR/depths" ]]; then
    N=$(ls "$DATA_DIR/images" | wc -l)
    echo "Already extracted ($N frames found at $DATA_DIR)."
    echo "Delete $DEST to re-download."
    exit 0
fi

mkdir -p "$DEST"

# ── resolve file ID from manifest ─────────────────────────────────────────────
echo "==> Fetching manifest …"
FILE_ID=$(curl -s "$BASE_URL/api/datasets/export?exporter=dataverse_json&persistentId=$PERSISTENT_ID" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = data['datasetVersion']['files']
for f in files:
    if f['dataFile']['filename'] == '$VIRTUAL_FILE':
        print(f['dataFile']['id'])
        break
")

if [[ -z "$FILE_ID" ]]; then
    echo "Error: could not find $VIRTUAL_FILE in dataset manifest."
    exit 1
fi
echo "    File ID: $FILE_ID"

# ── download ──────────────────────────────────────────────────────────────────
ARCHIVE="$DEST/$VIRTUAL_FILE"
echo "==> Downloading $VIRTUAL_FILE (~1.1 GB) …"
curl -L --progress-bar \
    "$BASE_URL/api/access/datafile/$FILE_ID" \
    -o "$ARCHIVE"

# ── extract ───────────────────────────────────────────────────────────────────
echo "==> Extracting …"
if command -v 7za &>/dev/null; then
    7za x "$ARCHIVE" -o"$DEST" -y
elif command -v 7zz &>/dev/null; then
    7zz x "$ARCHIVE" -o"$DEST" -y
else
    echo "    (7za/7zz not found — using py7zr)"
    python3 - <<PYEOF
import sys
try:
    import py7zr
except ImportError:
    print("Error: py7zr not installed. Run:  pip install py7zr", file=sys.stderr)
    sys.exit(1)
with py7zr.SevenZipFile("$ARCHIVE", mode="r") as z:
    z.extractall(path="$DEST")
PYEOF
fi

rm "$ARCHIVE"

# ── verify ────────────────────────────────────────────────────────────────────
N_IMG=$(ls "$DATA_DIR/images" | wc -l)
N_DEP=$(ls "$DATA_DIR/depths" | wc -l)
echo ""
echo "==> Done."
echo "    images : $N_IMG frames"
echo "    depths : $N_DEP depth maps"
echo "    dest   : $DEST"
echo ""
echo "Run a quick check:"
echo "  python -c \""
echo "    from endo_da3.data.polypsense3d import PolypSense3DVirtualDataset"
echo "    ds = PolypSense3DVirtualDataset('$DEST', split='train')"
echo "    print(len(ds), ds[0]['images'].shape)"
echo "  \""
