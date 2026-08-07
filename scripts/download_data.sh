#!/usr/bin/env bash
# Fetch the official warm-up data + checkpoint from the lab's server.
# Idempotent: wget -c resumes partial downloads; extraction skipped if present.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data
cd data

echo "[1/3] Demo CT scans"
wget -c --no-check-certificate http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz

echo "[2/3] Extracting"
if [ ! -d AbdomenAtlasDemo ]; then
    tar -xzvf AbdomenAtlasDemo.tar.gz
else
    echo "AbdomenAtlasDemo/ already extracted, skipping"
fi

echo "[3/3] Pretrained checkpoint (large file, resumable)"
wget -c --no-check-certificate http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth

echo
echo "Done."
echo "  CT scans   : $(pwd)/AbdomenAtlasDemo"
echo "  Checkpoint : $(pwd)/swin_unetr_totalsegmentator_vertebrae.pth"
find AbdomenAtlasDemo -name ct.nii.gz | sort
