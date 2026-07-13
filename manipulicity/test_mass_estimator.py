"""
test_mass_estimator.py
----------------------
Batch-test MassEstimator against every image found in a test folder.

Usage:
    # OpenAI (default)
    python test_mass_estimator.py --test_dir tests/images --api_key sk-...

    # Gemini
    python test_mass_estimator.py --test_dir tests/images --model gemini-1.5-flash --gemini_api_key AIza...

    # Local Ollama
    python test_mass_estimator.py --test_dir tests/images --model ollama/llama3.2-vision:11b

Optional flags:
    --save_json results.json   Save all results to a JSON file.
    --ground_truth gt.json     Compare against ground-truth masses.
                               Expected format: {"filename.jpg": <mass_kg>, ...}
    --stop_on_error            Abort on the first failed image (default: continue).
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from mass_estimator import MassEstimator

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_images(test_dir: str) -> list[Path]:
    """Return a sorted list of image paths found recursively under *test_dir*."""
    root = Path(test_dir)
    if not root.exists():
        raise FileNotFoundError(f"Test directory not found: {root.resolve()}")
    images = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    return images


def _print_separator(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _print_result(idx: int, total: int, image_path: Path, result: dict, elapsed: float) -> None:
    _print_separator()
    print(f"[{idx}/{total}]  {image_path.name}")
    objects = result.get("objects", [])
    if len(objects) > 1:
        print(f"  Objects found: {len(objects)}")
        for i, obj in enumerate(objects, start=1):
            occluded = " [occluded]" if obj.get("occluded") else ""
            print(f"  [{i}] {obj['object_description']}{occluded}")
            print(f"       Material : {obj['material_guess']}")
            print(
                f"       Mass     : {obj['mass_kg']:.4f} kg  "
                f"[{obj['mass_kg_range'][0]:.4f}, {obj['mass_kg_range'][1]:.4f}]  "
                f"({obj['confidence']})"
            )
        print(f"  Total Mass   : {result['mass_kg']:.4f} kg")
        print(
            f"  Total Range  : [{result['mass_kg_range'][0]:.4f}, "
            f"{result['mass_kg_range'][1]:.4f}] kg"
        )
    else:
        print(f"  Object       : {result['object_description']}")
        print(f"  Material     : {result['material_guess']}")
        print(f"  Mass         : {result['mass_kg']:.4f} kg")
        print(
            f"  Range        : [{result['mass_kg_range'][0]:.4f}, "
            f"{result['mass_kg_range'][1]:.4f}] kg"
        )
    print(f"  Confidence   : {result['confidence']}")
    print(f"  Reasoning    : {result['reasoning']}")
    print(f"  Time         : {elapsed:.2f} s")


def _print_summary(
    records: list[dict],
    ground_truth: Optional[dict[str, float]],
) -> None:
    _print_separator("=")
    print("SUMMARY")
    _print_separator("=")

    passed = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] == "error"]
    total = len(records)

    print(f"  Total images : {total}")
    print(f"  Succeeded    : {len(passed)}")
    print(f"  Failed       : {len(failed)}")

    if passed:
        avg_mass = sum(r["mass_kg"] for r in passed) / len(passed)
        avg_time = sum(r["elapsed_s"] for r in passed) / len(passed)
        print(f"  Avg mass     : {avg_mass:.4f} kg")
        print(f"  Avg time     : {avg_time:.2f} s")

    # Ground-truth comparison
    if ground_truth and passed:
        _print_separator()
        print("GROUND-TRUTH COMPARISON")
        _print_separator()
        abs_errors = []
        for r in passed:
            name = r["image"]
            if name in ground_truth:
                gt = ground_truth[name]
                pred = r["mass_kg"]
                abs_err = abs(pred - gt)
                rel_err = abs_err / gt * 100 if gt != 0 else float("inf")
                abs_errors.append(abs_err)
                print(
                    f"  {name:<30s}  GT={gt:.4f} kg  "
                    f"Pred={pred:.4f} kg  "
                    f"AbsErr={abs_err:.4f} kg  "
                    f"RelErr={rel_err:.1f}%"
                )
        if abs_errors:
            mae = sum(abs_errors) / len(abs_errors)
            print(f"\n  MAE over {len(abs_errors)} matched image(s): {mae:.4f} kg")

    if failed:
        _print_separator()
        print("FAILED IMAGES")
        for r in failed:
            print(f"  {r['image']}  →  {r['error']}")

    _print_separator("=")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-test MassEstimator on images in a test folder."
    )
    parser.add_argument(
        "--test_dir",
        default="tests/images",
        help="Directory containing test images (searched recursively). Default: tests/images.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help=(
            "VLM to use: an OpenAI model (e.g. gpt-4o-mini, gpt-4o), a "
            "Gemini model (e.g. gemini-1.5-flash), or a local model prefixed "
            "with 'ollama/' (e.g. ollama/llama3.2-vision:11b). "
            "Default: gpt-4o-mini."
        ),
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--gemini_api_key",
        default=None,
        help="Google API key for Gemini models (falls back to GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--local_base_url",
        default=None,
        help="Base URL of a local OpenAI-compatible server (e.g. http://localhost:11434/v1).",
    )
    parser.add_argument(
        "--save_json",
        default=None,
        metavar="PATH",
        help="If set, save all results to this JSON file.",
    )
    parser.add_argument(
        "--ground_truth",
        default=None,
        metavar="PATH",
        help=(
            'Path to a JSON file mapping filename → ground-truth mass (kg), '
            'e.g. {"apple.jpg": 0.182, "mug.png": 0.350}.'
        ),
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Abort immediately on the first failed image.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_arg_parser().parse_args()

    # Load ground truth if provided
    ground_truth: Optional[dict[str, float]] = None
    if args.ground_truth:
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    # Discover images
    try:
        images = _find_images(args.test_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not images:
        print(f"[WARNING] No images found in '{args.test_dir}'. "
              f"Supported extensions: {', '.join(sorted(_IMAGE_EXTENSIONS))}")
        sys.exit(0)

    print(f"[TestRunner] Found {len(images)} image(s) in '{args.test_dir}'")
    print(f"[TestRunner] Model: {args.model}\n")

    # Build estimator once (avoids re-authenticating per image)
    try:
        estimator = MassEstimator(
            api_key=args.api_key,
            model=args.model,
            gemini_api_key=args.gemini_api_key,
            local_base_url=args.local_base_url,
        )
    except (ValueError, ImportError) as exc:
        print(f"[ERROR] Failed to initialise MassEstimator: {exc}")
        sys.exit(1)

    records: list[dict] = []
    total = len(images)

    for idx, img_path in enumerate(images, start=1):
        t0 = time.perf_counter()
        try:
            result = estimator.estimate(img_path)
            elapsed = time.perf_counter() - t0
            _print_result(idx, total, img_path, result, elapsed)
            records.append({
                "image": img_path.name,
                "path": str(img_path),
                "status": "ok",
                "elapsed_s": round(elapsed, 3),
                **result,
            })
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            error_msg = str(exc)
            print(f"[{idx}/{total}]  {img_path.name}  →  ERROR: {error_msg}")
            if args.stop_on_error:
                traceback.print_exc()
                sys.exit(1)
            records.append({
                "image": img_path.name,
                "path": str(img_path),
                "status": "error",
                "elapsed_s": round(elapsed, 3),
                "error": error_msg,
            })

    # Summary
    print()
    _print_summary(records, ground_truth)

    # Save JSON
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\n[TestRunner] Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
