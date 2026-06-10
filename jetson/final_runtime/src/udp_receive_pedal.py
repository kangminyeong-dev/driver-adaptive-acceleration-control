#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from dataclasses import dataclass


@dataclass
class PedalMessage:
    timestamp: float
    pedal_percent: float
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive pedal input over UDP on the Jetson.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address. Use 0.0.0.0 to listen on all interfaces.")
    parser.add_argument("--port", type=int, default=5005, help="UDP port to listen on.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional socket timeout in seconds.")
    parser.add_argument("--max-messages", type=int, default=0, help="Stop after this many valid messages. 0 means forever.")
    return parser.parse_args()


def parse_message(text: str) -> PedalMessage:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty message")

    if stripped.startswith("{"):
        payload = json.loads(stripped)
        timestamp = float(payload.get("time", payload.get("t")))
        pedal_percent = float(payload.get("pedal_percent", payload.get("pedal")))
        return PedalMessage(timestamp=timestamp, pedal_percent=pedal_percent, raw=stripped)

    row = next(csv.reader([stripped]))
    if len(row) < 2:
        raise ValueError("expected 'time,pedal_percent'")
    return PedalMessage(timestamp=float(row[0]), pedal_percent=float(row[1]), raw=stripped)


def main() -> None:
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    if args.timeout is not None:
        sock.settimeout(args.timeout)

    print("[Jetson UDP pedal receiver]")
    print(f"- listening: {args.host}:{args.port}")
    print("- accepted formats:")
    print("  0.2,12.5")
    print('  {"time":0.2,"pedal_percent":12.5}')
    print("- stop: Ctrl+C")

    count = 0
    last_arrival = None
    try:
        while True:
            try:
                data, address = sock.recvfrom(4096)
            except TimeoutError:
                print("[timeout] no message received")
                break
            arrival = time.time()
            text = data.decode("utf-8", errors="replace")
            try:
                message = parse_message(text)
            except Exception as exc:
                print(f"[bad] from={address[0]}:{address[1]} raw={text.strip()!r} error={exc}")
                continue

            count += 1
            dt = 0.0 if last_arrival is None else arrival - last_arrival
            last_arrival = arrival
            print(
                f"[{count:05d}] from={address[0]}:{address[1]} "
                f"t={message.timestamp:.3f} pedal={message.pedal_percent:.2f}% "
                f"arrival_dt={dt:.3f}s"
            )
            if args.max_messages > 0 and count >= args.max_messages:
                break
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
