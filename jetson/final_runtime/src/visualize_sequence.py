#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from model_runtime import load_classifier, predict_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
STATE_COLORS = {
    "off": (70, 70, 70),
    "on": (70, 190, 80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record sequence inference visualization to an MP4 file.")
    parser.add_argument("input_dir", type=Path, help="Directory of sequential images.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "runtime_config.json",
        help="Runtime config path.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional torch device, for example cuda or cpu.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of dummy GPU/model warmup passes before timing.")
    parser.add_argument("--scale", type=float, default=0.55, help="Display/video scale for source images.")
    parser.add_argument("--delay-ms", type=int, default=1, help="Extra OpenCV wait time between displayed frames.")
    parser.add_argument("--loops", type=int, default=1, help="Number of sample-folder loops to record.")
    parser.add_argument("--show", action="store_true", help="Also show a live OpenCV window while recording.")
    parser.add_argument(
        "--record",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "visualization.mp4",
        help="Save the live visualization frames to an MP4 file.",
    )
    parser.add_argument("--no-record", action="store_true", help="Disable MP4 recording.")
    parser.add_argument("--record-fps", type=float, default=20.0, help="Playback FPS for the recorded MP4 file.")
    return parser.parse_args()


def image_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (10**9, path.name)


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=image_sort_key,
    )


def draw_label(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.75) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 8, y - text_h - 10), (x + text_w + 8, y + baseline + 8), color, -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_bar(
    frame,
    label: str,
    value: float,
    origin: tuple[int, int],
    width: int,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    clamped = max(0.0, min(1.0, value))
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (35, 35, 35), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * clamped), y + 14), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (160, 160, 160), 1)


def make_frame(
    image_path: Path,
    result: dict[str, float | str],
    index: int,
    total: int,
    cycle: int,
    frame_count: int,
    effective_state: str,
    pending_state: str | None,
    pending_count: int,
    switch_threshold: int,
    inference_ms: float,
    avg_ms: float,
    scale: float,
):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    h, w = frame.shape[:2]
    panel_h = 178
    canvas = cv2.copyMakeBorder(frame, panel_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))

    prediction = str(result["prediction"])
    pred_color = STATE_COLORS.get(prediction, (80, 120, 220))
    eff_color = STATE_COLORS.get(effective_state, (80, 120, 220))
    instant_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    draw_label(canvas, f"loop {cycle}  frame {frame_count}  image {index + 1}/{total}  {image_path.name}", (18, 34), (55, 55, 55), 0.68)
    draw_label(canvas, f"RAW: {prediction.upper()}  E={int(result['e_raw'])}", (18, 78), pred_color, 0.78)
    draw_label(canvas, f"EFFECTIVE: {effective_state.upper()}", (18, 122), eff_color, 0.78)

    right_x = max(360, int(w * 0.52))
    cv2.putText(
        canvas,
        f"inference {inference_ms:6.1f} ms   {instant_fps:5.1f} FPS",
        (right_x, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"average   {avg_ms:6.1f} ms   {avg_fps:5.1f} FPS",
        (right_x, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )

    pending_text = pending_state.upper() if pending_state else "-"
    cv2.putText(
        canvas,
        f"pending {pending_text} {pending_count}/{switch_threshold}",
        (right_x, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )

    bar_w = min(260, max(120, w - right_x - 24))
    draw_bar(canvas, "OFF", float(result["off_score"]), (right_x, 122), bar_w, STATE_COLORS["off"])
    draw_bar(canvas, "ON", float(result["on_score"]), (right_x, 152), bar_w, STATE_COLORS["on"])

    return canvas


def warmup_model(model, config: dict, device: torch.device, warmup_count: int) -> None:
    if warmup_count <= 0:
        return
    height = int(config["image_height"])
    width = int(config["image_width"])
    dummy = torch.zeros((1, 3, height, width), device=device)
    with torch.inference_mode():
        for _ in range(warmup_count):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")
    if args.loops <= 0:
        raise ValueError("--loops must be 1 or greater for MP4 recording.")

    model, config, device = load_classifier(args.config, args.device)
    warmup_model(model, config, device, args.warmup)

    switch_threshold = int(config["switch_threshold"])
    e_mapping = config.get("e_mapping", {})

    effective_state = None
    pending_state = None
    pending_count = 0
    durations: list[float] = []
    frame_count = 0
    cycle = 1
    running = True
    writer = None

    print("[시각화 영상 기록 시작]")
    print(f"- input_dir: {args.input_dir}")
    print(f"- images: {len(image_paths)}")
    print(f"- device: {device}")
    print(f"- loops: {args.loops}")
    if args.no_record:
        args.record = None

    if args.record:
        print(f"- recording: {args.record}")
    if args.show:
        print("- live window: enabled, press q or Esc to stop early")

    while running and cycle <= args.loops:
        for index, image_path in enumerate(image_paths):
            frame_count += 1
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = predict_image(model, config, device, image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - start) * 1000

            durations.append(inference_ms)
            recent = durations[-30:]
            avg_ms = sum(recent) / len(recent)

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

            frame = make_frame(
                image_path=image_path,
                result=result,
                index=index,
                total=len(image_paths),
                cycle=cycle,
                frame_count=frame_count,
                effective_state=str(effective_state),
                pending_state=pending_state,
                pending_count=pending_count,
                switch_threshold=switch_threshold,
                inference_ms=inference_ms,
                avg_ms=avg_ms,
                scale=args.scale,
            )

            if args.record and writer is None:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.record), fourcc, args.record_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {args.record}")

            if writer is not None:
                writer.write(frame)

            print(
                f"[loop {cycle} frame {frame_count}] "
                f"{image_path.name} raw={prediction.upper()} "
                f"effective={str(effective_state).upper()} "
                f"{inference_ms:.1f}ms {1000.0 / inference_ms if inference_ms > 0 else 0.0:.1f}FPS"
            )

            if args.show:
                cv2.imshow("Jetson live inference - press q to quit", frame)
                key = cv2.waitKey(max(1, args.delay_ms)) & 0xFF
                if key in {ord("q"), 27}:
                    running = False
                    break

        cycle += 1

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    avg_ms = sum(durations) / len(durations) if durations else 0.0
    print("[시각화 완료]")
    print(f"- device: {device}")
    print(f"- total_frames: {frame_count}")
    print(f"- avg_inference_ms: {avg_ms:.3f}")
    print(f"- avg_fps: {1000.0 / avg_ms if avg_ms > 0 else 0.0:.2f}")
    if args.record:
        print(f"- recorded_mp4: {args.record}")


if __name__ == "__main__":
    main()
