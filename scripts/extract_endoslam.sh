#!/usr/bin/env bash
# Extract EndoSLAM.zip and its nested scene zips.
#
# Usage:
#   bash scripts/extract_endoslam.sh \
#       --zip  /path/to/EndoSLAM.zip \
#       --dest /path/to/data/EndoSLAM
#
# Result layout:
#   <dest>/UnityCam/
#     Calibration/cam.txt
#     Colon/
#       Poses/colon_position_rotation.csv
#       Frames/image_NNNN.png          (~21 887 frames)
#       Pixelwise Depths/aov_image_NNNN.png
#     Small Intestine/ ...             (~12 558 frames)
#     Stomach/ ...                     (~1 548 frames)
#
# The outer zip is ~10 GB; the three Frames zips together are ~4 GB.

set -euo pipefail

ZIP=""
DEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip)  ZIP="$2";  shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$ZIP"  ]] && { echo "Error: --zip  required"; exit 1; }
[[ -z "$DEST" ]] && { echo "Error: --dest required"; exit 1; }

echo "==> Extracting outer zip to $DEST …"
mkdir -p "$DEST"
unzip -q -o "$ZIP" "UnityCam/*" -d "$DEST"

UNITYCAM="$DEST/UnityCam"

# ── per-scene nested zips ──────────────────────────────────────────────────
declare -A SCENES=(
    ["Colon"]="Colon"
    ["Small Intestine"]="Small Intestine"
    ["Stomach"]="Stomach"
)

for scene in "Colon" "Small Intestine" "Stomach"; do
    scene_dir="$UNITYCAM/$scene"
    echo ""
    echo "==> Scene: $scene"

    frames_zip="$scene_dir/Frames.zip"
    depths_zip="$scene_dir/Pixelwise Depths.zip"

    if [[ -f "$frames_zip" ]]; then
        echo "    Extracting Frames.zip …"
        unzip -q -o "$frames_zip" -d "$scene_dir"
        rm "$frames_zip"
    else
        echo "    Frames already extracted (no Frames.zip found)."
    fi

    if [[ -f "$depths_zip" ]]; then
        echo "    Extracting 'Pixelwise Depths.zip' …"
        unzip -q -o "$depths_zip" -d "$scene_dir"
        rm "$depths_zip"
    else
        echo "    Depths already extracted."
    fi
done

echo ""
echo "==> Done. Dataset layout:"
find "$UNITYCAM" -maxdepth 3 -name "*.csv" -o -name "cam.txt" | sort
echo ""
echo "Run a quick check:"
echo "  python -c \""
echo "    from endo_da3.data.endoslam import EndoSLAMSynthDataset"
echo "    ds = EndoSLAMSynthDataset('$DEST', split='train')"
echo "    print(len(ds), ds[0]['images'].shape)"
echo "  \""
