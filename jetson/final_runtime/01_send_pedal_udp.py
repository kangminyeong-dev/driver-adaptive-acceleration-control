import argparse
import csv
import socket
import time
from pathlib import Path


DEFAULT_CSV = "pedal_input_sequence_deadzone_pull_rounded_1dp_120s.csv"
DEFAULT_BROADCAST_IP = "255.255.255.255"
DEFAULT_PORT = 5005


def read_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"time_sec", "pedal_percent"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV header must include: {', '.join(sorted(required))}")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                time_sec = float(row["time_sec"])
                pedal_percent = float(row["pedal_percent"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value at CSV line {row_number}: {row}") from exc

            rows.append((time_sec, pedal_percent))

    if not rows:
        raise ValueError("CSV has no data rows.")

    return rows


def format_message(time_sec, pedal_percent):
    return f"{time_sec:g},{pedal_percent:g}"


def send_message(sock, address, time_sec, pedal_percent, verbose=True):
    message = format_message(time_sec, pedal_percent)
    sock.sendto(message.encode("utf-8"), address)
    if verbose:
        print(f"sent -> {address[0]}:{address[1]}  {message}")


def configure_socket(sock, broadcast=False):
    if broadcast:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)


def send_one(ip, port, time_sec, pedal_percent, broadcast=False):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        configure_socket(sock, broadcast)
        send_message(sock, (ip, port), time_sec, pedal_percent)


def replay_csv(csv_path, ip, port, speed, broadcast=False):
    rows = read_rows(csv_path)
    address = (ip, port)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        configure_socket(sock, broadcast)
        previous_time = None
        for time_sec, pedal_percent in rows:
            if previous_time is not None:
                delay = max(0.0, time_sec - previous_time) / speed
                time.sleep(delay)

            send_message(sock, address, time_sec, pedal_percent)
            previous_time = time_sec


def main():
    parser = argparse.ArgumentParser(
        description="Send pedal CSV rows to Jetson over UDP as 'time,pedal_percent'."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"CSV path. Default: {DEFAULT_CSV}")
    parser.add_argument("--ip", help="Jetson IP. If omitted, sends to LAN broadcast.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Jetson UDP port. Default: {DEFAULT_PORT}")
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help=f"Send to LAN broadcast address ({DEFAULT_BROADCAST_IP}). This is the default when --ip is omitted.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier. 1.0 means real CSV timing, 2.0 means twice as fast.",
    )
    parser.add_argument(
        "--one",
        action="store_true",
        help="Send only one test line. Uses --time and --pedal.",
    )
    parser.add_argument("--time", type=float, default=0.0, help="Time value for --one test message.")
    parser.add_argument("--pedal", type=float, default=0.0, help="Pedal percent value for --one test message.")

    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0.")

    use_broadcast = args.broadcast or args.ip is None
    target_ip = DEFAULT_BROADCAST_IP if use_broadcast else args.ip

    if args.one:
        send_one(target_ip, args.port, args.time, args.pedal, use_broadcast)
    else:
        replay_csv(args.csv, target_ip, args.port, args.speed, use_broadcast)


if __name__ == "__main__":
    main()
