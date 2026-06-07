"""Post-process F5 heatmaps: build index CSV + best-of composite sheet."""
from __future__ import annotations
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import re

heatmap_dir = Path(sys.argv[1])
files = sorted(heatmap_dir.glob("[0-9][0-9][0-9]_*.png"))
print(f"[INFO] {len(files)} heatmap files")

pattern = re.compile(r"^(\d+)_(.+?)_(correct_real|correct_fake|FP_predfake|FN_predreal)\.png$")
rows = []
for p in files:
    m = pattern.match(p.name)
    if not m:
        print(f"  skip: {p.name}")
        continue
    rows.append({
        "idx": int(m.group(1)),
        "file": p.name,
        "generator": m.group(2),
        "tag": m.group(3),
    })
pd.DataFrame(rows).to_csv(heatmap_dir / "_index.csv", index=False)
print(f"[SAVED] _index.csv ({len(rows)} rows)")

# Best-of: pick 6 samples (3 generators × 2 tag types)
chosen_files = []
for g in ["Stable_Diffusion_v1.4", "ADM", "GLIDE"]:
    for tag in ["correct_real", "correct_fake"]:
        match = next((r for r in rows if r["generator"] == g and r["tag"] == tag), None)
        if match:
            chosen_files.append(heatmap_dir / match["file"])

print(f"[INFO] composing best-of sheet from {len(chosen_files)} samples")
imgs = [Image.open(f) for f in chosen_files]
max_w = max(im.width for im in imgs)
sum_h = sum(im.height for im in imgs)
composite = Image.new("RGB", (max_w, sum_h), color="white")
y = 0
for im in imgs:
    composite.paste(im, ((max_w - im.width) // 2, y))
    y += im.height
composite.save(heatmap_dir / "_bestof_sheet.png")
print(f"[SAVED] _bestof_sheet.png  ({composite.size})")
