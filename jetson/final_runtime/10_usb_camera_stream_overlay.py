import argparse
import time

import cv2


DEFAULT_DEVICE = "/dev/video0"
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_FPS = 30
DEFAULT_UDP_PORT = 5000
WINDOW_NAME = "USB Camera Stream Overlay"


def parse_args():
    parser = argparse.ArgumentParser(description="Stream USB camera video with a large ON/OFF overlay.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Camera device path or index. Example: /dev/video0, /dev/video1, 0",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Requested capture FPS.")
    parser.add_argument("--window-width", type=int, default=960, help="Preview window width.")
    parser.add_argument("--window-height", type=int, default=540, help="Preview window height.")
    parser.add_argument("--windowed", action="store_true", help="Show in a normal window instead of fullscreen.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a Jetson preview window.")
    parser.add_argument("--flip", action="store_true", help="Flip the preview horizontally.")
    parser.add_argument("--udp-host", help="Laptop IP address to receive the video stream.")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="UDP video stream port.")
    parser.add_argument(
        "--state",
        type=int,
        choices=(0, 1),
        default=0,
        help="Fixed overlay state for testing. 1 shows ON, 0 shows OFF.",
    )
    return parser.parse_args()


def normalize_device(device):
    if str(device).isdigit():
        return int(device)
    return device


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(normalize_device(device), cv2.CAP_V4L2)
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


def draw_status_overlay(frame, state):
    label = "ON" if state == 1 else "OFF"
    color = (0, 220, 0) if state == 1 else (0, 0, 255)
    height, width = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.6, height / 190.0)
    thickness = max(3, int(height / 150))
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    x = (width - text_w) // 2
    y = height - max(28, int(height * 0.055))

    cv2.putText(frame, label, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_preview_status(frame, device, fps_value):
    text = f"{device} | stream overlay | {fps_value:4.1f} FPS | q/ESC quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 120),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    cap = open_camera(args.device, args.width, args.height, args.fps)
    udp_writer = open_udp_stream(args.udp_host, args.udp_port, args.width, args.height, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened {args.device}: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    print(f"Overlay state: {'ON' if args.state == 1 else 'OFF'}")
    if udp_writer is not None:
        print(f"Streaming video to udp://{args.udp_host}:{args.udp_port}")

    if not args.no_display:
        print("Press q or ESC in the preview window to quit.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if args.windowed:
            cv2.resizeWindow(WINDOW_NAME, args.window_width, args.window_height)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        print("Running without Jetson preview window. Press Ctrl+C to quit.")

    frames = 0
    fps_value = 0.0
    last_time = time.monotonic()
    preview_width = args.window_width if args.windowed else args.width
    preview_height = args.window_height if args.windowed else args.height

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            stream_frame = resize_to_16_9(frame, args.width, args.height)
            draw_status_overlay(stream_frame, args.state)

            if udp_writer is not None:
                udp_writer.write(stream_frame)

            display_frame = stream_frame
            if not args.no_display and (preview_width != args.width or preview_height != args.height):
                display_frame = cv2.resize(stream_frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

            frames += 1
            now = time.monotonic()
            elapsed = now - last_time
            if elapsed >= 1.0:
                fps_value = frames / elapsed
                frames = 0
                last_time = now

            if args.no_display:
                continue

            draw_preview_status(display_frame, args.device, fps_value)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        if udp_writer is not None:
            udp_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
