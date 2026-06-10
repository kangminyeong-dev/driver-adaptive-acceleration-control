#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from PIL import Image

from model_runtime import load_classifier, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential inference with effective-state switching.")
    parser.add_argument("input_dir", type=Path, help="Directory of sequential images.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "runtime_config.json",
        help="Runtime config path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sequence_inference.csv",
        help="Where to write sequence inference results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device, for example cuda or cpu.",
    )
    return parser.parse_args()


def image_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (10**9, path.name)


def main() -> None:
    args = parse_args()
    model, config, device = load_classifier(args.config, args.device)
    switch_threshold = int(config["switch_threshold"])
    e_mapping = config.get("e_mapping", {})

    image_paths = sorted(
        [p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=image_sort_key,
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    effective_state = None
    pending_state = None
    pending_count = 0
    rows: list[dict[str, object]] = []
    inference_durations: list[float] = []

    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        inference_start = time.perf_counter()
        result = predict_image(model, config, device, image)
        inference_elapsed = time.perf_counter() - inference_start
        inference_durations.append(inference_elapsed)
        prediction = str(result["prediction"])

        if effective_state is None:
            effective_state = prediction
            pending_state = None
            pending_count = 0
        elif prediction == effective_state:
            pending_state = None
            pending_count = 0
        else:
            if prediction == pending_state:
                pending_count += 1
            else:
                pending_state = prediction
                pending_count = 1

            if pending_count >= switch_threshold:
                effective_state = prediction
                pending_state = None
                pending_count = 0

        rows.append(
            {
                "index": index,
                "image_name": image_path.name,
                "prediction": prediction,
                "E_raw": int(e_mapping.get(prediction, 0)),
                "effective_state": effective_state,
                "E_effective": int(e_mapping.get(effective_state, 0)),
                "inference_ms": f"{inference_elapsed * 1000:.3f}",
                "confidence": f"{result['confidence']:.6f}",
                "off_score": f"{result['off_score']:.6f}",
                "on_score": f"{result['on_score']:.6f}",
                "pending_state": pending_state or "",
                "pending_count": pending_count,
            }
        )

        print(
            f"[{index + 1}/{len(image_paths)}] "
            f"image={image_path.name} "
            f"pred={prediction.upper()} "
            f"E_raw={int(e_mapping.get(prediction, 0))} "
            f"E_effective={int(e_mapping.get(effective_state, 0))} "
            f"inference_ms={inference_elapsed * 1000:.3f}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image_name",
                "prediction",
                "E_raw",
                "effective_state",
                "E_effective",
                "inference_ms",
                "confidence",
                "off_score",
                "on_score",
                "pending_state",
                "pending_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    avg_inference_ms = (
        sum(inference_durations) / len(inference_durations) * 1000
        if inference_durations
        else 0.0
    )

    print("[시퀀스 추론 완료]")
    print(f"- input_dir: {args.input_dir}")
    print(f"- total_images: {len(rows)}")
    print(f"- switch_threshold: {switch_threshold}")
    print(f"- avg_inference_ms: {avg_inference_ms:.3f}")
    print(f"- output_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
