import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_runtime import load_classifier, predict_image


DEFAULT_CONFIG = ROOT_DIR / "config" / "runtime_config.json"
DEFAULT_DEVICE = "/dev/video0"
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_FPS = 30
DEFAULT_UDP_PORT = 5000
WINDOW_NAME = "USB Camera Live Environment Inference"
STATE_COLORS = {
    "off": (0, 0, 255),
    "on": (0, 220, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run realtime environment inference from a USB camera.")
    parser.add_argument("--camera-device", default=DEFAULT_DEVICE, help=f"USB camera device. Default: {DEFAULT_DEVICE}")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Camera/stream width. Default: {DEFAULT_WIDTH}")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Camera/stream height. Default: {DEFAULT_HEIGHT}")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Camera/stream FPS. Default: {DEFAULT_FPS}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"Runtime config path. Default: {DEFAULT_CONFIG}")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--warmup", type=int, default=2, help="Model warmup passes. Default: 2")
    parser.add_argument("--infer-every", type=int, default=1, help="Run inference every N camera frames. Default: 1")
    parser.add_argument(
        "--switch-threshold",
        type=int,
        default=50,
        help="Consecutive opposite predictions needed to switch the effective state. Default: 50",
    )
    parser.add_argument("--windowed", action="store_true", help="Show in a normal window instead of fullscreen.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a Jetson preview window.")
    parser.add_argument("--window-width", type=int, default=960, help="Preview window width. Default: 960")
    parser.add_argument("--window-height", type=int, default=540, help="Preview window height. Default: 540")
    parser.add_argument("--flip", action="store_true", help="Flip the camera frame horizontally.")
    parser.add_argument("--udp-host", help="Optional laptop IP address to receive visualized RTP/JPEG video.")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help=f"UDP video port. Default: {DEFAULT_UDP_PORT}")
    return parser.parse_args()


def normalize_camera_device(device):
    if str(device).isdigit():
        return int(device)
    return device


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(normalize_camera_device(device), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_udp_stream(host, port, width, height, fps):
    if not host:
        return None

    pipeline = (
        f"appsrc ! videoconvert ! video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 "
        f"! jpegenc quality=70 ! rtpjpegpay ! udpsink host={host} port={port} sync=false async=false"
    )
    writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError("Cannot open UDP video stream. Check OpenCV GStreamer support.")
    return writer


def resize_to_16_9(frame, width, height):
    target_ratio = 16.0 / 9.0
    frame_h, frame_w = frame.shape[:2]
    frame_ratio = frame_w / frame_h

    if frame_ratio > target_ratio:
        crop_w = int(frame_h * target_ratio)
        x0 = (frame_w - crop_w) // 2
        frame = frame[:, x0 : x0 + crop_w]
    elif frame_ratio < target_ratio:
        crop_h = int(frame_w / target_ratio)
        y0 = (frame_h - crop_h) // 2
        frame = frame[y0 : y0 + crop_h, :]

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def warmup_model(model, config, device, warmup_count):
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


def frame_to_pil_rgb(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def update_effective_state(raw_prediction, effective_state, pending_state, pending_count, switch_threshold):
    if effective_state is None:
        return raw_prediction, None, 0

    if raw_prediction == effective_state:
        return effective_state, None, 0

    if raw_prediction == pending_state:
        pending_count += 1
    else:
        pending_state = raw_prediction
        pending_count = 1

    if pending_count >= switch_threshold:
        return raw_prediction, None, 0

    return effective_state, pending_state, pending_count


def draw_label(frame, text, origin, color, scale=0.75):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x - 8, y - text_h - 10), (x + text_w + 8, y + baseline + 8), color, -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_bar(frame, label, value, origin, width, color):
    x, y = origin
    value = max(0.0, min(1.0, float(value)))
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (35, 35, 35), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * value), y + 14), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (160, 160, 160), 1)


def draw_inference_overlay(
    frame,
    result,
    effective_state,
    pending_state,
    pending_count,
    switch_threshold,
    inference_ms,
    camera_fps,
):
    raw_prediction = str(result["prediction"])
    raw_color = STATE_COLORS.get(raw_prediction, (80, 120, 220))
    effective_color = STATE_COLORS.get(str(effective_state), (80, 120, 220))
    height, width = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (width, 128), (18, 18, 18), -1)
    draw_label(frame, f"RAW: {raw_prediction.upper()}  E={int(result['e_raw'])}", (18, 38), raw_color, 0.78)
    draw_label(frame, f"EFFECTIVE: {str(effective_state).upper()}", (18, 86), effective_color, 0.78)

    right_x = max(360, int(width * 0.52))
    cv2.putText(
        frame,
        f"inference {inference_ms:6.1f} ms   camera {camera_fps:5.1f} FPS",
        (right_x, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    pending_text = pending_state.upper() if pending_state else "-"
    cv2.putText(
        frame,
        f"pending {pending_text} {pending_count}/{switch_threshold}",
        (right_x, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )

    bar_w = min(280, max(140, width - right_x - 24))
    draw_bar(frame, "OFF", result["off_score"], (right_x, 88), bar_w, STATE_COLORS["off"])
    draw_bar(frame, "ON", result["on_score"], (right_x, 116), bar_w, STATE_COLORS["on"])

    label = "ON" if str(effective_state).lower() == "on" else "OFF"
    color = STATE_COLORS.get(str(effective_state).lower(), (0, 0, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.6, height / 190.0)
    thickness = max(3, int(height / 150))
    (text_w, _), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x = (width - text_w) // 2
    y = height - max(28, int(height * 0.055))
    cv2.putText(frame, label, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    if args.infer_every <= 0:
        raise ValueError("--infer-every must be 1 or greater.")

    model, config, device = load_classifier(args.config, args.device)
    warmup_model(model, config, device, args.warmup)
    switch_threshold = int(args.switch_threshold)
    if switch_threshold <= 0:
        raise ValueError("--switch-threshold must be 1 or greater.")

    cap = open_camera(args.camera_device, args.width, args.height, args.fps)
    udp_writer = open_udp_stream(args.udp_host, args.udp_port, args.width, args.height, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened {args.camera_device}: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    print(f"Model: {args.config}")
    print(f"Device: {device}")
    print(f"Input size: {config['image_width']}x{config['image_height']}")
    if udp_writer is not None:
        print(f"Streaming visualized video to udp://{args.udp_host}:{args.udp_port}")
    if args.no_display:
        print("Display disabled. Press Ctrl+C to stop.")
    else:
        print("Press q or ESC in the preview window to quit.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(WINDOW_NAME, args.window_width, args.window_height)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    result = {
        "prediction": "off",
        "e_raw": 0,
        "confidence": 0.0,
        "off_score": 0.0,
        "on_score": 0.0,
    }
    effective_state = None
    pending_state = None
    pending_count = 0
    inference_ms = 0.0
    frame_count = 0
    camera_frames = 0
    camera_fps = 0.0
    fps_time = time.monotonic()
    preview_width = args.window_width if args.windowed else args.width
    preview_height = args.window_height if args.windowed else args.height

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame_count += 1
            camera_frames += 1
            if args.flip:
                frame = cv2.flip(frame, 1)

            visual_frame = resize_to_16_9(frame, args.width, args.height)

            if frame_count % args.infer_every == 0:
                image = frame_to_pil_rgb(visual_frame)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = predict_image(model, config, device, image)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - start) * 1000.0

                raw_prediction = str(result["prediction"]).lower()
                effective_state, pending_state, pending_count = update_effective_state(
                    raw_prediction,
                    effective_state,
                    pending_state,
                    pending_count,
                    switch_threshold,
                )

            now = time.monotonic()
            elapsed = now - fps_time
            if elapsed >= 1.0:
                camera_fps = camera_frames / elapsed
                camera_frames = 0
                fps_time = now

            if effective_state is None:
                effective_state = str(result["prediction"]).lower()
            draw_inference_overlay(
                visual_frame,
                result,
                effective_state,
                pending_state,
                pending_count,
                switch_threshold,
                inference_ms,
                camera_fps,
            )

            if udp_writer is not None:
                udp_writer.write(visual_frame)

            if args.no_display:
                continue

            display_frame = visual_frame
            if preview_width != args.width or preview_height != args.height:
                display_frame = cv2.resize(visual_frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        if udp_writer is not None:
            udp_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
