#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from model_runtime import load_classifier, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference with the final classifier.")
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "runtime_config.json",
        help="Runtime config path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device, for example cuda or cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config, device = load_classifier(args.config, args.device)

    with Image.open(args.image) as img:
        image = img.convert("RGB")

    result = predict_image(model, config, device, image)

    print("[단일 이미지 추론 결과]")
    print(f"- image: {args.image}")
    print(f"- device: {device}")
    print(f"- prediction: {result['prediction']}")
    print(f"- E_raw: {result['e_raw']}")
    print(f"- confidence: {result['confidence']:.4f}")
    print(f"- off_score: {result['off_score']:.4f}")
    print(f"- on_score: {result['on_score']:.4f}")


if __name__ == "__main__":
    main()
