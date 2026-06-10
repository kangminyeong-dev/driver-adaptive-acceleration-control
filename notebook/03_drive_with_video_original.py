# python .\main.py --ip 10.82.166.7


import argparse
import io
import subprocess
import sys
import threading
import time

from PIL import Image

from receive_jetson_video import DEFAULT_PORT as DEFAULT_VIDEO_PORT
from receive_jetson_video import DEFAULT_SINK, RTP_JPEG_CAPS, build_pipeline, find_gst_launch
from run import (
    DEFAULT_CORRECTED_PORT,
    DEFAULT_GRAPH_WINDOW,
    DEFAULT_INTERVAL,
    DEFAULT_IP,
    DEFAULT_PORT,
    LivePedalUdpPlot,
)


def start_video_receiver(port, sink, fullscreen, quiet):
    gst_launch = find_gst_launch()
    pipeline = build_pipeline(port, sink, fullscreen)

    if not gst_launch:
        print("Warning: GStreamer gst-launch-1.0 was not found.")
        print("Video receiver was not started, but UDP sending will continue.")
        print("Expected command:")
        print("  gst-launch-1.0 " + " ".join(pipeline))
        return None

    command = [gst_launch, *pipeline]
    if not quiet:
        print("Starting Jetson video receiver:")
        print("  " + " ".join(command))

    try:
        return subprocess.Popen(command)
    except OSError as exc:
        print(f"Warning: failed to start video receiver: {exc}")
        print("UDP sending will continue.")
        return None


def stop_video_receiver(process):
    if process is None or process.poll() is not None:
        return

    print("Stopping video receiver...")
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3.0)


class EmbeddedVideoReceiver:
    def __init__(self, port, quiet):
        self.port = port
        self.quiet = quiet
        self.process = None
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False

    def start(self):
        gst_launch = find_gst_launch()
        if not gst_launch:
            print("Warning: GStreamer gst-launch-1.0 was not found.")
            print("Embedded stream area will stay black.")
            return False

        command = [
            gst_launch,
            "-q",
            "udpsrc",
            f"port={self.port}",
            f"caps={RTP_JPEG_CAPS}",
            "!",
            "rtpjpegdepay",
            "!",
            "jpegparse",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]
        if not self.quiet:
            print("Starting embedded Jetson video receiver:")
            print("  " + " ".join(command))

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            print(f"Warning: failed to start embedded video receiver: {exc}")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _read_stdout(self):
        buffer = bytearray()
        stdout = self.process.stdout
        if stdout is None:
            return

        while self.running:
            chunk = stdout.read(4096)
            if not chunk:
                break

            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(buffer) > 1024 * 1024:
                        del buffer[:-2]
                    break

                if start > 0:
                    del buffer[:start]

                end = buffer.find(b"\xff\xd9", 2)
                if end < 0:
                    break

                jpeg_bytes = bytes(buffer[: end + 2])
                del buffer[: end + 2]

                try:
                    frame = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                except OSError:
                    continue

                with self.lock:
                    self.latest_frame = frame


class IntegratedPedalUdpPlot(LivePedalUdpPlot):
    def __init__(self, *args, video_process=None, embedded_video=None, **kwargs):
        self.video_process = video_process
        self.embedded_video = embedded_video
        self._video_exit_reported = False
        super().__init__(*args, **kwargs)

    def get_stream_frame(self):
        if self.embedded_video is None:
            return None
        return self.embedded_video.get_frame()

    def update(self):
        if self.video_process is not None and not self._video_exit_reported:
            exit_code = self.video_process.poll()
            if exit_code is not None:
                print(f"Warning: video receiver exited with code {exit_code}. UDP sending continues.")
                self._video_exit_reported = True

        super().update()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send Windows pedal input to Jetson and receive Jetson RTP/JPEG video."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"Jetson receiver IP. Default: {DEFAULT_IP}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Jetson pedal UDP port. Default: {DEFAULT_PORT}")
    parser.add_argument(
        "--corrected-port",
        type=int,
        default=DEFAULT_CORRECTED_PORT,
        help=f"Corrected pedal UDP receive port. Default: {DEFAULT_CORRECTED_PORT}",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Pedal send interval in seconds. Default: {DEFAULT_INTERVAL:g}",
    )
    parser.add_argument(
        "--graph-window",
        type=float,
        default=DEFAULT_GRAPH_WINDOW,
        help=f"Pedal graph time range in seconds. Default: {DEFAULT_GRAPH_WINDOW:g}",
    )
    parser.add_argument("--joystick-id", type=int, default=0, help="Windows joystick ID. Default: 0")
    parser.add_argument("--quiet", action="store_true", help="Send UDP packets without printing every packet.")

    parser.add_argument(
        "--external-video",
        action="store_true",
        help="Also start the old separate GStreamer video receiver window.",
    )
    parser.add_argument("--no-video", action="store_true", help="Compatibility option. The dashboard video area stays black.")
    parser.add_argument(
        "--video-port",
        type=int,
        default=DEFAULT_VIDEO_PORT,
        help=f"Local UDP port for Jetson video. Default: {DEFAULT_VIDEO_PORT}",
    )
    parser.add_argument(
        "--video-sink",
        default=DEFAULT_SINK,
        help=f"GStreamer video sink. Default: {DEFAULT_SINK}. Try autovideosink if needed.",
    )
    parser.add_argument("--video-windowed", action="store_true", help="Show video in a window instead of fullscreen.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0.")
    if args.graph_window <= 0:
        raise ValueError("--graph-window must be greater than 0.")

    video_process = None
    embedded_video = None
    if args.external_video and not args.no_video:
        video_process = start_video_receiver(
            args.video_port,
            args.video_sink,
            fullscreen=not args.video_windowed,
            quiet=args.quiet,
        )
        if video_process is not None:
            time.sleep(0.5)
    elif not args.no_video:
        embedded_video = EmbeddedVideoReceiver(args.video_port, args.quiet)
        if embedded_video.start():
            time.sleep(0.5)
        else:
            embedded_video = None

    try:
        IntegratedPedalUdpPlot(
            args.ip,
            args.port,
            args.interval,
            args.joystick_id,
            not args.quiet,
            args.corrected_port,
            args.graph_window,
            video_process=video_process,
            embedded_video=embedded_video,
        ).run()
    finally:
        if embedded_video is not None:
            embedded_video.stop()
        stop_video_receiver(video_process)


if __name__ == "__main__":
    sys.exit(main())
