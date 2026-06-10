#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PedalSample:
    source_time: float
    pedal_percent: float
    arrival_time: float


@dataclass
class PedalEvent:
    event_id: int
    start_time: float
    end_time: float
    peak_time: float
    peak_pedal: float
    max_slope: float
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize UDP pedal input and pedal events in real time.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=5005, help="UDP port to listen on.")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Visible time range in seconds.")
    parser.add_argument("--start-threshold", type=float, default=5.0, help="Pedal percent that starts an event.")
    parser.add_argument("--end-threshold", type=float, default=5.0, help="Pedal percent that ends an event.")
    parser.add_argument("--log-csv", type=Path, default=Path("pedal_udp_events.csv"), help="Event summary CSV path.")
    parser.add_argument("--record", type=Path, default=Path("pedal_udp_visualization.mp4"), help="Save graph visualization to MP4.")
    parser.add_argument("--no-record", action="store_true", help="Disable MP4 recording.")
    parser.add_argument("--record-fps", type=float, default=20.0, help="Playback FPS for the recorded MP4.")
    return parser.parse_args()


def parse_message(text: str) -> tuple[float, float]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty message")
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        return float(payload.get("time", payload.get("t"))), float(payload.get("pedal_percent", payload.get("pedal")))
    row = next(csv.reader([stripped]))
    if len(row) < 2:
        raise ValueError("expected 'time,pedal_percent'")
    return float(row[0]), float(row[1])


def event_summary(event_id: int, samples: list[PedalSample]) -> PedalEvent:
    if len(samples) < 2:
        sample = samples[0]
        return PedalEvent(event_id, sample.source_time, sample.source_time, sample.source_time, sample.pedal_percent, 0.0, 1)

    peak = max(samples, key=lambda sample: sample.pedal_percent)
    max_slope = 0.0
    for prev, curr in zip(samples, samples[1:]):
        dt = curr.source_time - prev.source_time
        if dt <= 0:
            continue
        max_slope = max(max_slope, (curr.pedal_percent - prev.pedal_percent) / dt)

    return PedalEvent(
        event_id=event_id,
        start_time=samples[0].source_time,
        end_time=samples[-1].source_time,
        peak_time=peak.source_time,
        peak_pedal=peak.pedal_percent,
        max_slope=max_slope,
        sample_count=len(samples),
    )


def append_event_log(path: Path, event: PedalEvent) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "start_time",
                "end_time",
                "duration",
                "peak_time",
                "peak_pedal",
                "max_slope",
                "sample_count",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "event_id": event.event_id,
                "start_time": f"{event.start_time:.3f}",
                "end_time": f"{event.end_time:.3f}",
                "duration": f"{event.end_time - event.start_time:.3f}",
                "peak_time": f"{event.peak_time:.3f}",
                "peak_pedal": f"{event.peak_pedal:.3f}",
                "max_slope": f"{event.max_slope:.3f}",
                "sample_count": event.sample_count,
            }
        )


def main() -> None:
    args = parse_args()
    if args.no_record:
        args.record = None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.setblocking(False)

    samples: deque[PedalSample] = deque()
    events: list[PedalEvent] = []
    active_event_samples: list[PedalSample] = []
    in_event = False
    event_id = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 5))
    line, = ax.plot([], [], color="#1f77b4", linewidth=2, label="pedal")
    current_marker, = ax.plot([], [], "o", color="#d62728", markersize=7, label="current")
    peak_markers, = ax.plot([], [], "^", color="#ff7f0e", markersize=9, linestyle="none", label="event peak")
    ax.axhline(args.start_threshold, color="#2ca02c", linestyle="--", linewidth=1, label="start threshold")
    ax.axhline(args.end_threshold, color="#7f7f7f", linestyle=":", linewidth=1, label="end threshold")
    ax.set_title("UDP Pedal Input / Event Detection")
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("pedal (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    event_spans = []
    last_status = ""
    writer = None
    stop_requested = False

    def on_key_press(event) -> None:
        nonlocal stop_requested
        if event.key in {"q", "escape"}:
            stop_requested = True

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    print("[UDP pedal graph]")
    print(f"- listening: {args.host}:{args.port}")
    print(f"- event: pedal > {args.start_threshold}% starts, pedal <= {args.end_threshold}% ends")
    if args.record:
        print(f"- recording: {args.record}")
    print("- stop/save: press q, close graph window, or Ctrl+C")

    try:
        while plt.fignum_exists(fig.number) and not stop_requested:
            received = False
            while True:
                try:
                    data, address = sock.recvfrom(4096)
                except BlockingIOError:
                    break
                received = True
                text = data.decode("utf-8", errors="replace")
                try:
                    source_time, pedal_percent = parse_message(text)
                except Exception as exc:
                    print(f"[bad] raw={text.strip()!r} error={exc}")
                    continue

                sample = PedalSample(source_time=source_time, pedal_percent=pedal_percent, arrival_time=time.time())
                samples.append(sample)

                while samples and sample.source_time - samples[0].source_time > args.window_sec:
                    samples.popleft()

                if not in_event and pedal_percent > args.start_threshold:
                    in_event = True
                    event_id += 1
                    active_event_samples = [sample]
                    print(f"[event {event_id}] start t={source_time:.3f} pedal={pedal_percent:.2f}%")
                elif in_event:
                    active_event_samples.append(sample)
                    if pedal_percent <= args.end_threshold:
                        in_event = False
                        event = event_summary(event_id, active_event_samples)
                        events.append(event)
                        append_event_log(args.log_csv, event)
                        print(
                            f"[event {event.event_id}] end duration={event.end_time - event.start_time:.3f}s "
                            f"peak={event.peak_pedal:.2f}% max_slope={event.max_slope:.2f}%/s"
                        )
                        active_event_samples = []

                last_status = f"last from {address[0]}:{address[1]}  t={source_time:.2f}  pedal={pedal_percent:.2f}%"

            if samples:
                xs = [sample.source_time for sample in samples]
                ys = [sample.pedal_percent for sample in samples]
                line.set_data(xs, ys)
                current_marker.set_data([xs[-1]], [ys[-1]])

                x_min = max(xs[-1] - args.window_sec, xs[0])
                x_max = max(xs[-1] + 1.0, x_min + args.window_sec)
                ax.set_xlim(x_min, x_max)

                peak_x = [event.peak_time for event in events if x_min <= event.peak_time <= x_max]
                peak_y = [event.peak_pedal for event in events if x_min <= event.peak_time <= x_max]
                peak_markers.set_data(peak_x, peak_y)

                for span in event_spans:
                    span.remove()
                event_spans = []
                for event in events:
                    if event.end_time < x_min or event.start_time > x_max:
                        continue
                    event_spans.append(ax.axvspan(event.start_time, event.end_time, color="#ffbb78", alpha=0.25))
                if in_event and active_event_samples:
                    event_spans.append(ax.axvspan(active_event_samples[0].source_time, xs[-1], color="#98df8a", alpha=0.22))

            ax.set_title(f"UDP Pedal Input / Event Detection    {last_status}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if args.record:
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                image = np.asarray(fig.canvas.buffer_rgba())
                frame = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                if writer is None:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(args.record), fourcc, args.record_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {args.record}")
                writer.write(frame)

            plt.pause(0.03 if received else 0.08)
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        if writer is not None:
            writer.release()
            print(f"[recorded] {args.record}")
        sock.close()


if __name__ == "__main__":
    main()
