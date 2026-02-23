#!/bin/bash
# Download a representative C3VD v2 subset for Stage 1 training.
#
# Split strategy (segment-level, frozen before any training):
#
#   train : c1 t1 v1+v2 + c2 t1 v1+v2                     = 20 sequences
#           segments: ascending, transverse1, descending, sigmoid1, sigmoid2
#           (c2 has no sigmoid2; c2 descending starts at t3 → use t3)
#   val   : c1 t1 v1 + c2 t1 v1 (held-out segments)       =  6 sequences
#           segments: cecum, rectum, transverse2
#           (different 3D shapes from train — tests shape generalisation)
#   test  : c0 v1 (CC-BY-NC-SA 4.0, unseen phantom)        =  4 sequences
#           segments: cecum, transverse, sigmoid, full
#
# Excluded:
#   v3 — debris GT is from the clean version (inaccurate for depth training)
#   v4 — deformation videos, no pixel-level GT
#   t2/t3/t4 — same geometry as t1, different texture; irrelevant for Stage 1
#   _mold / _model — 3D meshes
#
# Total: 30 sequences
#
# Depth  : depth/NNNN_depth.tiff  16-bit [0,65535] → [0,100] mm
# RGB    : rgb/NNNN.png           1350×1080 px
# Pose   : pose.txt               per-frame row-major 4×4 c2w (metres)
# Intrin : camera_intrinsics.txt  shared across all sequences
#
# Licenses:
#   c0 : CC-BY-NC-SA 4.0 (non-commercial)
#   c1/c2 : CC-BY 4.0

set -e

DOI="10.7281/T1/JC64MK"
BASE_URL="https://archive.data.jhu.edu"
OUT_DIR="${1:-/home/in4218/code/data/C3VD}"

mkdir -p "$OUT_DIR/train" "$OUT_DIR/val" "$OUT_DIR/test"

TRAIN_FILES=(
    # c1 — train segments, t1, v1 + v2
    "c1_ascending_t1_v1.zip"
    "c1_ascending_t1_v2.zip"
    "c1_transverse1_t1_v1.zip"
    "c1_transverse1_t1_v2.zip"
    "c1_descending_t1_v1.zip"
    "c1_descending_t1_v2.zip"
    "c1_sigmoid1_t1_v1.zip"
    "c1_sigmoid1_t1_v2.zip"
    "c1_sigmoid2_t1_v1.zip"
    "c1_sigmoid2_t1_v2.zip"
    # c2 — train segments, t1 v1+v2 (descending: t3 only, c2 has no sigmoid2)
    "c2_ascending_t1_v1.zip"
    "c2_ascending_t1_v2.zip"
    "c2_transverse1_t1_v1.zip"
    "c2_transverse1_t1_v2.zip"
    "c2_descending_t3_v1.zip"
    "c2_descending_t3_v2.zip"
    "c2_sigmoid_t1_v1.zip"
    "c2_sigmoid_t1_v2.zip"
)

VAL_FILES=(
    # c1 — held-out segments (different shapes: wide pouch, narrow tube, sharp bend)
    "c1_cecum_t1_v1.zip"
    "c1_rectum_t1_v1.zip"
    "c1_transverse2_t1_v1.zip"
    # c2 — same held-out segments
    "c2_cecum_t1_v1.zip"
    "c2_rectum_t1_v1.zip"
    "c2_transverse2_t1_v1.zip"
)

TEST_FILES=(
    # c0 — C3VDv1, completely unseen phantom, CC-BY-NC-SA 4.0
    # note: c0 has no ascending/descending video sequences
    "c0_cecum_t1_v1.zip"
    "c0_transverse_t1_v1.zip"
    "c0_sigmoid_t1_v1.zip"
    "c0_full_t1_v1.zip"
)

# ── fetch manifest once ───────────────────────────────────────────────────────
echo "Fetching file manifest..."
MANIFEST=$(curl -s "$BASE_URL/api/datasets/:persistentId?persistentId=doi:$DOI")

download_file() {
    local filename="$1"
    local dest_dir="$2"
    local seq_name="${filename%.zip}"
    local dest="$dest_dir/$filename"

    if [ -d "$dest_dir/$seq_name" ]; then
        echo "  [skip] $seq_name already extracted"
        return
    fi

    local file_id
    file_id=$(echo "$MANIFEST" | jq -r ".data.latestVersion.files[] | select(.label==\"$filename\") | .dataFile.id")

    if [ -z "$file_id" ]; then
        echo "  [warn] $filename not found in manifest"
        return
    fi

    echo "  Downloading $filename (id=$file_id)..."
    local redirect
    redirect=$(curl -s -D - -o /dev/null "$BASE_URL/api/access/datafile/$file_id" \
               | grep -i '^Location: ' | cut -d' ' -f2 | tr -d '\r')
    curl -L -o "$dest" "$redirect"
    echo "  Extracting..."
    mkdir -p "$dest_dir/$seq_name"
    unzip -q -o "$dest" -d "$dest_dir/$seq_name"
    rm "$dest"
    echo "  Done: $seq_name"
}

echo "=== TRAIN (18 sequences) ==="
for f in "${TRAIN_FILES[@]}"; do download_file "$f" "$OUT_DIR/train"; done

echo "=== VAL (6 sequences — held-out segments) ==="
for f in "${VAL_FILES[@]}"; do download_file "$f" "$OUT_DIR/val"; done

echo "=== TEST (4 sequences — unseen phantom, CC-BY-NC-SA) ==="
for f in "${TEST_FILES[@]}"; do download_file "$f" "$OUT_DIR/test"; done

# Shared camera intrinsics
echo "=== Camera intrinsics ==="
INT_ID=$(echo "$MANIFEST" | jq -r '.data.latestVersion.files[] | select(.label=="camera_intrinsics.txt") | .dataFile.id')
redirect=$(curl -s -D - -o /dev/null "$BASE_URL/api/access/datafile/$INT_ID" \
           | grep -i '^Location: ' | cut -d' ' -f2 | tr -d '\r')
curl -sL -o "$OUT_DIR/camera_intrinsics.txt" "$redirect"
echo "  Saved camera_intrinsics.txt"

echo ""
echo "Done. $OUT_DIR/{train,val,test}/ — 28 sequences total."
echo "Train segments : ascending, transverse1, descending, sigmoid1, sigmoid2"
echo "Val segments   : cecum, rectum, transverse2  (held-out shapes)"
echo "Test phantom   : c0 (unseen, CC-BY-NC-SA)"
