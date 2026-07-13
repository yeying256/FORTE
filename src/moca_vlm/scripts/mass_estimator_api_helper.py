#!/usr/bin/env python3

import argparse
import base64
import io
import json
import sys
from pathlib import Path


def _default_vendor_python_path() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "python"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the manipulicity mass estimator in a dedicated Python environment."
    )
    parser.add_argument("--manipulicity_repo_path", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument(
        "--image_base64_stdin",
        action="store_true",
        help="Read base64-encoded image bytes from stdin instead of --image.",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--gemini_api_key", default="")
    parser.add_argument("--local_base_url", default="")
    parser.add_argument(
        "--vendor_python_path",
        default="",
        help=(
            "Optional directory containing vendored Python dependencies. "
            "Defaults to moca_vlm/vendor/python when that directory exists."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    repo_path = Path(args.manipulicity_repo_path).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"manipulicity repo was not found at: {repo_path}")

    vendor_path = (
        Path(args.vendor_python_path).expanduser().resolve()
        if args.vendor_python_path
        else _default_vendor_python_path()
    )
    if vendor_path.exists():
        sys.path.insert(0, str(vendor_path))

    sys.path.insert(0, str(repo_path))
    from mass_estimator import MassEstimator  # pylint: disable=import-error
    from PIL import Image

    kwargs = {"model": args.model}
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.gemini_api_key:
        kwargs["gemini_api_key"] = args.gemini_api_key
    if args.local_base_url:
        kwargs["local_base_url"] = args.local_base_url

    estimator = MassEstimator(**kwargs)
    if args.image_base64_stdin:
        raw_text = sys.stdin.read().strip()
        if not raw_text:
            raise ValueError("--image_base64_stdin was set but stdin was empty.")
        image_bytes = base64.b64decode(raw_text)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = estimator.estimate(image)
    else:
        if not args.image:
            raise ValueError("Either --image or --image_base64_stdin must be provided.")
        result = estimator.estimate(args.image)
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
