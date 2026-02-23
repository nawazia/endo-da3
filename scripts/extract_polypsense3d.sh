#!/usr/bin/env bash
# Extract Virtual-Dataset-For-PolypSense3D.7z to <dest>/PolypSense3D/.
#
# Usage:
#   bash scripts/extract_polypsense3d.sh \
#       --7z   /path/to/Virtual-Dataset-For-PolypSense3D.7z \
#       --dest /path/to/data/PolypSense3D
#
# Result layout:
#   <dest>/Virtual Dataset For PolypSense3D/depth_estimation/
#       camera.txt                   9 space-separated floats (3×3 K, row-major)
#       position_rotation.csv        (tX,tY,tZ,rX,rY,rZ,rW,time)
#       images/image_NNNN.jpg        320×320 RGB  (~8 241 frames)
#       depths/aov_image_NNNN.png    320×320 RGBA uint8 (R = depth in mm)
#
# Requires either p7zip (7za) or Python ≥3.8 with py7zr installed.
#   Install p7zip :  sudo apt install p7zip-full
#   Install py7zr :  pip install py7zr

set -euo pipefail

ARCHIVE=""
DEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --7z)   ARCHIVE="$2"; shift 2 ;;
        --dest) DEST="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$ARCHIVE" ]] && { echo "Error: --7z  required"; exit 1; }
[[ -z "$DEST"    ]] && { echo "Error: --dest required"; exit 1; }
[[ -f "$ARCHIVE" ]] || { echo "Error: file not found: $ARCHIVE"; exit 1; }

mkdir -p "$DEST"

echo "==> Extracting $ARCHIVE → $DEST …"

if command -v 7za &>/dev/null; then
    7za x "$ARCHIVE" -o"$DEST" -y
elif command -v 7zz &>/dev/null; then
    7zz x "$ARCHIVE" -o"$DEST" -y
else
    echo "    (7za/7zz not found — falling back to py7zr)"
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

DATA_DIR="$DEST/Virtual Dataset For PolypSense3D/depth_estimation"

echo ""
echo "==> Done. Dataset layout:"
find "$DATA_DIR" -maxdepth 1 | sort
echo "    images/ : $(ls "$DATA_DIR/images" | wc -l) frames"
echo "    depths/ : $(ls "$DATA_DIR/depths" | wc -l) depth maps"
echo ""
echo "Run a quick check:"
echo "  python -c \""
echo "    from endo_da3.data.polypsense3d import PolypSense3DVirtualDataset"
echo "    ds = PolypSense3DVirtualDataset('$DEST', split='train')"
echo "    print(len(ds), ds[0]['images'].shape)"
echo "  \""
