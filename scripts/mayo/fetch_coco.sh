#!/usr/bin/env bash
# Stage COCO 2017 (official zips) under the cf-transfer data root. Idempotent: skips finished files.
set -euo pipefail
ROOT=/rodata/azradonc_dev/m253405/cf-transfer/data/coco
mkdir -p "$ROOT/zips"; cd "$ROOT/zips"
for f in annotations_trainval2017.zip val2017.zip train2017.zip; do
  if [ -f "$f.done" ]; then echo "skip $f"; continue; fi
  echo "[$(date -u +%FT%TZ)] downloading $f"
  curl -sS -L -C - -o "$f" "http://images.cocodataset.org/zips/$f" || curl -sS -L -C - -o "$f" "http://images.cocodataset.org/annotations/$f"
  unzip -tq "$f" >/dev/null && touch "$f.done"
  echo "[$(date -u +%FT%TZ)] done $f $(stat -c %s "$f") bytes"
done
cd "$ROOT" && unzip -oq zips/annotations_trainval2017.zip 'annotations/instances_train2017.json' 'annotations/instances_val2017.json'
echo COCO_FETCH_DONE
