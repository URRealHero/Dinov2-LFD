#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal render test for pyrender-based pipeline.

- Generates N spherical views (default 10)
- Renders with encode.render_views_to_tempdir(...)
- Runs quick sanity checks on the resulting PNGs
- Produces a contact_sheet.png and a report.json
- Returns exit code 0 on pass, 1 on failure

Usage:
  python test_render_views.py /path/to/model.glb \
      --views 10 --res 512 --out ./render_test_out --cleanup

Notes:
  - Assumes PYOPENGL_PLATFORM=egl (prints a warning if not).
  - Uses your encode.generate_spherical_views / render_views_to_tempdir.
"""

import os
import sys
import math
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

try:
    import numpy as np
    from PIL import Image, ImageOps
except Exception as e:
    print("❌ Missing dependencies. Please install: numpy, pillow")
    raise

# Import your module that contains generate_spherical_views and render_views_to_tempdir
import encode


def compute_image_metrics(img: Image.Image) -> Dict[str, float]:
    """
    Very lightweight quality heuristics:
    - per-image std (higher => more content, not flat)
    - fraction of pixels that are not close to the light gray background (~244)
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    std = float(arr.std())

    # Count pixels that are not near the background.
    # Your bg ~ 0xF4 (244). Treat anything < 250 as "not-bg".
    non_bg = np.any(arr < 250, axis=2)
    non_bg_ratio = float(non_bg.mean())

    return {"std": std, "non_bg_ratio": non_bg_ratio}


def make_contact_sheet(image_paths: List[Path], dest: Path, thumb: int = 256) -> None:
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    # Preserve aspect by fitting into thumb x thumb
    thumbs = [ImageOps.contain(im, (thumb, thumb)) for im in imgs]

    n = len(thumbs)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    margin = 8

    sheet_w = cols * thumb + (cols + 1) * margin
    sheet_h = rows * thumb + (rows + 1) * margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), (240, 240, 240))
    for idx, im in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = margin + c * (thumb + margin)
        y = margin + r * (thumb + margin)
        sheet.paste(im, (x, y))
    sheet.save(dest)


def main():
    parser = argparse.ArgumentParser(description="Quick render sanity test with 10 views.")
    parser.add_argument("model", type=str, help="Path to mesh (e.g., .glb/.obj/.ply)")
    parser.add_argument("--views", type=int, default=10, help="Number of views to render")
    parser.add_argument("--res", type=int, default=512, help="Image resolution (square)")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: ./render_test_OUT/<stem>_<ts>)")
    parser.add_argument("--cleanup", action="store_true", help="Remove per-view PNGs after contact sheet/report")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    ts = time.strftime("%Y%m%d-%H%M%S")
    base_out = Path(args.out) if args.out else Path("render_test_OUT") / f"{model_path.stem}_{ts}"
    base_out.mkdir(parents=True, exist_ok=True)

    # 1) Generate views
    print(f"➡️  Generating {args.views} spherical views …")
    views = encode.generate_spherical_views(num_views=args.views)

    # 2) Render with your pyrender-based function
    print(f"➡️  Rendering to tempdir (res={args.res}) …")
    tmp_dir = encode.render_views_to_tempdir(str(model_path), views, resolution=args.res)

    if tmp_dir is None or not Path(tmp_dir).exists():
        print("❌ Rendering failed (no image directory returned). "
              "If you’re on a headless server, ensure EGL works: export PYOPENGL_PLATFORM=egl")
        sys.exit(1)

    # 3) Collect PNGs and run checks
    pngs = sorted(Path(tmp_dir).glob("view_*.png"))
    if len(pngs) == 0:
        print("❌ No PNGs produced.")
        sys.exit(1)

    # Copy images to final output for inspection
    copied_pngs = []
    for p in pngs:
        dest = base_out / p.name
        dest.write_bytes(p.read_bytes())
        copied_pngs.append(dest)

    print(f"✅ Rendered {len(copied_pngs)}/{args.views} images -> {base_out}")

    # 4) Simple quality metrics
    per_img = []
    blank_flags = 0
    for p in copied_pngs:
        try:
            im = Image.open(p)
            metrics = compute_image_metrics(im)
            per_img.append({"file": p.name, **metrics})
            # Heuristic “blank/flat” threshold:
            # std < 2.0 and non_bg_ratio < 0.01 means almost entirely flat bg
            if metrics["std"] < 2.0 and metrics["non_bg_ratio"] < 0.01:
                blank_flags += 1
        except Exception as e:
            per_img.append({"file": p.name, "error": str(e)})

    mean_std = float(np.mean([d["std"] for d in per_img if "std" in d])) if per_img else 0.0
    mean_non_bg = float(np.mean([d["non_bg_ratio"] for d in per_img if "non_bg_ratio" in d])) if per_img else 0.0

    # 5) Contact sheet
    sheet_path = base_out / "contact_sheet.png"
    try:
        make_contact_sheet(copied_pngs, sheet_path, thumb=min(256, args.res))
        print(f"🖼️  Contact sheet: {sheet_path}")
    except Exception as e:
        print(f"⚠️  Contact sheet failed: {e}")

    # 6) Report
    report = {
        "model_path": str(model_path),
        "output_dir": str(base_out),
        "requested_views": args.views,
        "rendered_images": len(copied_pngs),
        "blank_like_images": blank_flags,
        "mean_std": round(mean_std, 4),
        "mean_non_bg_ratio": round(mean_non_bg, 4),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM", None),
        "passed": (len(copied_pngs) == args.views and blank_flags == 0),
        "notes": [
            "Heuristics: std<2 AND non_bg_ratio<0.01 ⇒ likely blank/flat.",
            "Background assumed ≈ light gray; non-bg counts pixels < 250.",
        ],
        "per_image": per_img,
    }
    report_path = base_out / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"📝 Report: {report_path}")

    # 7) Optional cleanup of per-view PNGs (keep sheet + report)
    if args.cleanup:
        for p in copied_pngs:
            try:
                p.unlink()
            except Exception:
                pass
        try:
            # Also remove the temporary directory created by the renderer
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        print("🧹 Cleaned up individual PNGs. Kept contact_sheet.png and report.json.")

    # 8) Exit code
    if not report["passed"]:
        print("❌ Render test FAILED (see report.json).")
        sys.exit(1)
    print("✅ Render test PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
