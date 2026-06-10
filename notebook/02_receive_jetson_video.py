import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PORT = 5000
DEFAULT_SINK = "d3d11videosink"
RTP_JPEG_CAPS = (
    "application/x-rtp,"
    "media=(string)video,"
    "encoding-name=(string)JPEG,"
    "payload=(int)26,"
    "clock-rate=(int)90000"
)


def find_gst_launch():
    found = shutil.which("gst-launch-1.0")
    if found:
        return found

    candidates = [
        Path(r"C:\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"),
        Path(r"C:\gstreamer\1.0\mingw_x86_64\bin\gst-launch-1.0.exe"),
        Path(r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"),
        Path(r"C:\Program Files\gstreamer\1.0\mingw_x86_64\bin\gst-launch-1.0.exe"),
        Path(r"C:\Program Files (x86)\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "gstreamer" / "1.0" / "msvc_x86_64" / "bin" / "gst-launch-1.0.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "gstreamer" / "1.0" / "mingw_x86_64" / "bin" / "gst-launch-1.0.exe",
        Path(os.environ.get("GSTREAMER_1_0_ROOT_MSVC_X86_64", "")) / "bin" / "gst-launch-1.0.exe",
        Path(os.environ.get("GSTREAMER_1_0_ROOT_MINGW_X86_64", "")) / "bin" / "gst-launch-1.0.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def build_pipeline(port, sink, fullscreen):
    pipeline = [
        "udpsrc",
        f"port={port}",
        f"caps={RTP_JPEG_CAPS}",
        "!",
        "rtpjpegdepay",
        "!",
        "jpegdec",
        "!",
        "videoconvert",
        "!",
        sink,
    ]
    if fullscreen and sink in {"d3d11videosink", "direct3d11videosink"}:
        pipeline.append("fullscreen=true")
    pipeline.append("sync=false")
    return pipeline


def print_install_help():
    print("GStreamer gst-launch-1.0을 찾지 못했습니다.\n")
    print("Windows 설치 방법:")
    print("1. https://gstreamer.freedesktop.org/download/ 에서 Windows x86_64 MSVC Runtime 설치")
    print("2. 같은 페이지에서 Windows x86_64 MSVC Development도 함께 설치 권장")
    print(r"3. C:\gstreamer\1.0\msvc_x86_64\bin 을 PATH에 추가")
    print("\n설치 확인:")
    print("  gst-launch-1.0 --version")
    print("\nGStreamer 없이 빠른 대체 방법:")
    print("  VLC 실행 -> 미디어 -> 네트워크 스트림 열기 -> rtp://@:5000")
    print("단, RTP/JPEG payload 26 스트림은 GStreamer가 가장 확실합니다.")


def main():
    parser = argparse.ArgumentParser(
        description="Receive Jetson RTP/JPEG UDP video and show it fullscreen on Windows."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"UDP receive port. Default: {DEFAULT_PORT}")
    parser.add_argument(
        "--sink",
        default=DEFAULT_SINK,
        help="GStreamer video sink. Default: d3d11videosink. Try autovideosink if the screen stays blank.",
    )
    parser.add_argument("--windowed", action="store_true", help="Do not request fullscreen mode.")
    parser.add_argument("--print-only", action="store_true", help="Print the gst-launch command without running it.")
    args = parser.parse_args()

    gst_launch = find_gst_launch()
    pipeline = build_pipeline(args.port, args.sink, not args.windowed)

    if not gst_launch:
        print_install_help()
        print("\n설치 후 실행할 명령:")
        print("  gst-launch-1.0 " + " ".join(pipeline))
        return 1

    command = [gst_launch, *pipeline]
    print("Running:")
    print("  " + " ".join(command))
    print("\nStop: Ctrl+C")

    if args.print_only:
        return 0

    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
