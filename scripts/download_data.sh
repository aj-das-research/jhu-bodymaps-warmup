#!/usr/bin/env bash
# Fetch the official warm-up data + checkpoint from the lab's server.
# Idempotent: resumes partial downloads; extraction skipped if present.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data
cd data

# Prefer wget; fall back to curl (macOS often lacks wget).
download() {
    local url="$1"
    local out
    out="$(basename "$url")"
    if command -v wget >/dev/null 2>&1; then
        wget -c --no-check-certificate -O "$out" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -k -o "$out" "$url"
    else
        echo "error: need wget or curl to download $url" >&2
        exit 1
    fi
}

echo "[1/3] Demo CT scans"
download http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz

echo "[2/3] Extracting"
if [ ! -d AbdomenAtlasDemo ]; then
    tar -xzvf AbdomenAtlasDemo.tar.gz
else
    echo "AbdomenAtlasDemo/ already extracted, skipping"
fi

echo "[3/3] Pretrained checkpoint (large file, resumable)"
download http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth

echo
echo "Done."
echo "  CT scans   : $(pwd)/AbdomenAtlasDemo"
echo "  Checkpoint : $(pwd)/swin_unetr_totalsegmentator_vertebrae.pth"
find AbdomenAtlasDemo -name ct.nii.gz | sort
