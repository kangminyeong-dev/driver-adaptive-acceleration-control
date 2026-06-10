#python .\run.py --ip 10.249.246.7


import argparse
import ctypes
import socket
import threading
import time
import tkinter as tk
from collections import deque

from PIL import ImageTk


DEFAULT_IP = "10.249.246.7"
DEFAULT_PORT = 5005
DEFAULT_CORRECTED_PORT = 5006
DEFAULT_INTERVAL = 0.01
DEFAULT_GRAPH_WINDOW = 10.0

JOY_RETURNX = 0x00000001
JOY_RETURNY = 0x00000002
JOY_RETURNBUTTONS = 0x00000080
JOY_RETURNALL = JOY_RETURNX | JOY_RETURNY | JOY_RETURNBUTTONS

STEERING_CENTER = 32767.5
STEERING_DEADZONE = 1500
STEERING_MIN = 0
STEERING_MAX = 65535

PEDAL_CENTER = 32767.5
BRAKE_ON_THRESHOLD = 40000
BRAKE_OFF_THRESHOLD = 37000

MODE_BUTTON = 10
EMERGENCY_BUTTON = 9


class JOYINFOEX(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint),
        ("dwFlags", ctypes.c_uint),
        ("dwXpos", ctypes.c_uint),
        ("dwYpos", ctypes.c_uint),
        ("dwZpos", ctypes.c_uint),
        ("dwRpos", ctypes.c_uint),
        ("dwUpos", ctypes.c_uint),
        ("dwVpos", ctypes.c_uint),
        ("dwButtons", ctypes.c_uint),
        ("dwButtonNumber", ctypes.c_uint),
        ("dwPOV", ctypes.c_uint),
        ("dwReserved1", ctypes.c_uint),
        ("dwReserved2", ctypes.c_uint),
    ]


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def read_input_state(joystick_id=0):
    info = JOYINFOEX()
    info.dwSize = ctypes.sizeof(JOYINFOEX)
    info.dwFlags = JOY_RETURNALL

    result = ctypes.windll.winmm.joyGetPosEx(joystick_id, ctypes.byref(info))
    if result != 0:
        return None
    return info


def is_button_down(button_bits, button_number):
    return bool(button_bits & (1 << (button_number - 1)))


def x_to_steering(raw_x):
    delta = raw_x - STEERING_CENTER
    if abs(delta) <= STEERING_DEADZONE:
        return 0

    if delta < 0:
        usable_range = STEERING_CENTER - STEERING_DEADZONE - STEERING_MIN
        steering = (delta + STEERING_DEADZONE) / usable_range * 100
    else:
        usable_range = STEERING_MAX - STEERING_CENTER - STEERING_DEADZONE
        steering = (delta - STEERING_DEADZONE) / usable_range * 100

    return int(round(clamp(steering, -100, 100)))


def y_to_accel_0_100(y_value):
    if y_value >= PEDAL_CENTER:
        return 0.0

    accel = (PEDAL_CENTER - y_value) / PEDAL_CENTER * 100
    return max(0.0, min(100.0, accel))


def y_to_brake(raw_y, previous_brake):
    if previous_brake:
        return 0 if raw_y <= BRAKE_OFF_THRESHOLD else 1
    return 1 if raw_y >= BRAKE_ON_THRESHOLD else 0


def format_message(time_sec, pedal_percent, steering, brake, mode, emergency):
    return f"{time_sec:.2f},{pedal_percent:.1f},{steering},{brake},{mode},{emergency}"


class LivePedalUdpPlot:
    def __init__(
        self,
        ip,
        port,
        interval,
        joystick_id,
        verbose,
        corrected_port=DEFAULT_CORRECTED_PORT,
        graph_window=DEFAULT_GRAPH_WINDOW,
    ):
        self.ip = ip
        self.port = port
        self.corrected_port = corrected_port
        self.address = (ip, port)
        self.interval_ms = max(1, int(interval * 1000))
        self.joystick_id = joystick_id
        self.verbose = verbose
        self.window_seconds = graph_window
        self.start_time = time.perf_counter()
        self.samples = deque()
        self.corrected_samples = deque()
        self.corrected_lock = threading.Lock()
        self.corrected_time_offset = None
        self.latest_corrected = None
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.corrected_sock = self.open_corrected_socket(corrected_port)
        self.corrected_thread = None
        self.previous_input = None
        self.mode = 1
        self.emergency = 0
        self.brake = 0

        self.root = tk.Tk()
        self.root.title("Driving Dashboard")

        self.width = 1280
        self.height = 720
        self.panel_padding = 18
        self.graph_margin_left = 52
        self.graph_margin_right = 22
        self.graph_margin_top = 64
        self.graph_margin_bottom = 44

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#071923", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.minsize(900, 520)
        self.root.state("zoomed")
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.fullscreen = False

        print(f"Sending live pedal UDP to {ip}:{port}")
        if self.corrected_sock is not None:
            print(f"Receiving corrected pedal UDP on 0.0.0.0:{corrected_port}")
        else:
            print(f"Warning: corrected pedal UDP port {corrected_port} is not available.")
        print("Format: time_sec,pedal_percent,steering,brake,mode,emergency")
        print(f"Button {MODE_BUTTON}: mode 1 -> 2 -> 3 -> 4, emergency resets to 0")
        print(f"Button {EMERGENCY_BUTTON}: emergency 0 -> 1 -> 0")
        print("Close the dashboard window or press Ctrl+C to stop.")
        print("F11: fullscreen toggle, Esc: exit fullscreen.")

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.start_corrected_receiver()
        self.update()

    def open_corrected_socket(self, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.settimeout(0.2)
            return sock
        except OSError as exc:
            print(f"Warning: failed to bind corrected pedal UDP {port}: {exc}")
            return None

    def start_corrected_receiver(self):
        if self.corrected_sock is None:
            return
        self.corrected_thread = threading.Thread(target=self.receive_corrected_loop, daemon=True)
        self.corrected_thread.start()

    def receive_corrected_loop(self):
        while self.running:
            try:
                data, _address = self.corrected_sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                text = data.decode("utf-8").strip()
                remote_time_text, corrected_text = text.split(",", 1)
                remote_time = float(remote_time_text)
                corrected_value = clamp(float(corrected_text), 0.0, 100.0)
            except (UnicodeDecodeError, ValueError):
                continue

            local_elapsed = time.perf_counter() - self.start_time
            with self.corrected_lock:
                if self.corrected_time_offset is None:
                    self.corrected_time_offset = local_elapsed - remote_time

                plot_time = remote_time + self.corrected_time_offset
                self.corrected_samples.append((plot_time, corrected_value))
                self.latest_corrected = corrected_value

                cutoff = local_elapsed - self.window_seconds - 1.0
                while self.corrected_samples and self.corrected_samples[0][0] < cutoff:
                    self.corrected_samples.popleft()

    def on_canvas_resize(self, event):
        self.width = max(event.width, 900)
        self.height = max(event.height, 520)

    def toggle_fullscreen(self, _event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def exit_fullscreen(self, _event=None):
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)

    def close(self):
        self.running = False
        self.sock.close()
        if self.corrected_sock is not None:
            self.corrected_sock.close()
        self.root.destroy()

    def layout(self):
        right_width = max(300, int(self.width * 0.28))
        split_x = self.width - right_width
        split_y = int(self.height * 0.46)
        return {
            "stream": (0, 0, split_x, self.height),
            "graph_panel": (split_x, 0, self.width, split_y),
            "info_panel": (split_x, split_y, self.width, self.height),
        }

    def plot_area(self):
        x0, y0, x1, y1 = self.layout()["graph_panel"]
        return (
            x0 + self.graph_margin_left,
            y0 + self.graph_margin_top,
            x1 - self.graph_margin_right,
            y1 - self.graph_margin_bottom,
        )

    def point_to_canvas(self, t_value, pedal_percent, now):
        x0, y0, x1, y1 = self.plot_area()
        left_t = max(0.0, now - self.window_seconds)

        x = x0 + ((t_value - left_t) / self.window_seconds) * (x1 - x0)
        y = y1 - (pedal_percent / 100.0) * (y1 - y0)
        return x, y

    def draw_dashboard(self, now, pedal, steering, brake, mode, emergency, current_input):
        self.canvas.delete("all")
        layout = self.layout()
        stream_x0, stream_y0, stream_x1, stream_y1 = layout["stream"]
        graph_x0, graph_y0, graph_x1, graph_y1 = layout["graph_panel"]
        info_x0, info_y0, info_x1, info_y1 = layout["info_panel"]

        self.canvas.create_rectangle(stream_x0, stream_y0, stream_x1, stream_y1, fill="#05070a", outline="#122635", width=2)
        self.canvas.create_rectangle(graph_x0, graph_y0, graph_x1, graph_y1, fill="#102f3d", outline="#122635", width=2)
        self.canvas.create_rectangle(info_x0, info_y0, info_x1, info_y1, fill="#0d2632", outline="#122635", width=2)

        self.draw_stream_area(stream_x0, stream_y0, stream_x1, stream_y1)

        self.draw_graph_grid(now, pedal)
        self.draw_line(now)
        self.draw_drive_info(now, pedal, steering, brake, mode, emergency, current_input)

    def get_stream_frame(self):
        return None

    def draw_stream_area(self, x0, y0, x1, y1):
        frame = self.get_stream_frame()
        if frame is None:
            self.canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text="STREAM",
                fill="#334155",
                font=("Consolas", 36, "bold"),
            )
            return

        area_width = max(1, x1 - x0)
        area_height = max(1, y1 - y0)
        frame_width, frame_height = frame.size
        scale = min(area_width / frame_width, area_height / frame_height)
        display_size = (
            max(1, int(frame_width * scale)),
            max(1, int(frame_height * scale)),
        )
        display_frame = frame.resize(display_size)
        self.stream_photo = ImageTk.PhotoImage(display_frame)
        self.canvas.create_image(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            image=self.stream_photo,
            anchor="center",
        )

    def draw_graph_grid(self, now, current_value):
        x0, y0, x1, y1 = self.plot_area()
        panel_x0, panel_y0, panel_x1, _panel_y1 = self.layout()["graph_panel"]
        corrected = self.get_latest_corrected()

        self.canvas.create_text(
            panel_x0 + self.panel_padding,
            panel_y0 + 14,
            text="Pedal Monitor",
            fill="#f8fafc",
            anchor="nw",
            font=("Consolas", 15, "bold"),
        )

        legend_y = panel_y0 + 38
        self.canvas.create_line(
            panel_x0 + self.panel_padding,
            legend_y + 6,
            panel_x0 + self.panel_padding + 34,
            legend_y + 6,
            fill="#38bdf8",
            width=4,
        )
        self.canvas.create_text(
            panel_x0 + self.panel_padding + 42,
            legend_y,
            text="RAW",
            fill="#cbd5e1",
            anchor="nw",
            font=("Consolas", 9, "bold"),
        )
        self.canvas.create_line(
            panel_x0 + self.panel_padding + 92,
            legend_y + 6,
            panel_x0 + self.panel_padding + 126,
            legend_y + 6,
            fill="#ef4444",
            width=3,
        )
        self.canvas.create_text(
            panel_x0 + self.panel_padding + 134,
            legend_y,
            text="CORR",
            fill="#cbd5e1",
            anchor="nw",
            font=("Consolas", 9, "bold"),
        )

        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#3b5563", width=1)

        for value in range(0, 101, 20):
            y = y1 - (value / 100.0) * (y1 - y0)
            self.canvas.create_line(x0, y, x1, y, fill="#21404f")
            self.canvas.create_text(
                x0 - 10,
                y,
                text=str(value),
                fill="#cbd5e1",
                anchor="e",
                font=("Consolas", 10),
            )

        left_t = max(0.0, now - self.window_seconds)
        tick_count = 5
        for i in range(tick_count + 1):
            ratio = i / tick_count
            t_value = left_t + ratio * self.window_seconds
            x = x0 + ratio * (x1 - x0)
            self.canvas.create_line(x, y0, x, y1, fill="#21404f")
            self.canvas.create_text(
                x,
                y1 + 20,
                text=f"{t_value:.0f}s",
                fill="#cbd5e1",
                font=("Consolas", 11),
            )

    def draw_drive_info(self, now, pedal, steering, brake, mode, emergency, current_input):
        x0, y0, x1, y1 = self.layout()["info_panel"]
        pad = self.panel_padding
        line_y = y0 + 44
        line_gap = 28
        value_x = x0 + pad + 118

        steer_text = "CENTER"
        if steering < 0:
            steer_text = f"LEFT {abs(steering)}"
        elif steering > 0:
            steer_text = f"RIGHT {steering}"

        mode_names = {
            1: "\uc2e4\uc2dc\uac04",
            2: "\uace8\ubaa9",
            3: "\uc2dc\uac00\uc9c0",
            4: "\uace0\uc18d\ub3c4\ub85c",
        }
        corrected = self.get_latest_corrected()

        self.canvas.create_text(
            x0 + pad,
            y0 + 16,
            text="DRIVE INFO",
            fill="#f8fafc",
            anchor="nw",
            font=("Consolas", 16, "bold"),
        )

        rows = [
            ("UDP", f"{self.ip}:{self.port}", "#e2e8f0"),
            ("TIME", f"{now:.2f}s", "#e2e8f0"),
            ("PEDAL", f"{pedal:.1f}%", "#e2e8f0"),
            ("CORR", "--.-%" if corrected is None else f"{corrected:.1f}%", "#ef4444"),
            ("STEER", steer_text, "#e2e8f0"),
            ("BRAKE", "ON" if brake else "OFF", "#22c55e" if brake else "#ef4444"),
            ("MODE", mode_names.get(mode, "UNKNOWN"), "#e2e8f0"),
            ("EMERGENCY", "ON" if emergency else "OFF", "#facc15" if emergency else "#ef4444"),
        ]

        for label, value, value_color in rows:
            self.canvas.create_text(
                x0 + pad,
                line_y,
                text=f"{label:<10}",
                fill="#e2e8f0",
                anchor="nw",
                font=("Consolas", 13),
            )
            self.canvas.create_text(
                value_x,
                line_y,
                text=value,
                fill=value_color,
                anchor="nw",
                font=("Consolas", 13, "bold") if label in ("BRAKE", "EMERGENCY", "CORR") else ("Consolas", 13),
            )
            line_y += line_gap

        if current_input is not None:
            raw_text = f"RAW X={current_input.dwXpos} Y={current_input.dwYpos}"
            self.canvas.create_text(
                x0 + pad,
                y1 - 34,
                text=raw_text,
                fill="#94a3b8",
                anchor="nw",
                font=("Consolas", 11),
            )

        message_preview = format_message(now, pedal, steering, brake, mode, emergency)
        self.canvas.create_text(
            x0 + pad,
            y1 - 58,
            text=message_preview,
            fill="#38bdf8",
            anchor="nw",
            font=("Consolas", 11, "bold"),
        )

    def get_latest_corrected(self):
        with self.corrected_lock:
            return self.latest_corrected

    def get_corrected_samples(self):
        with self.corrected_lock:
            return list(self.corrected_samples)

    def draw_line(self, now):
        corrected_points = []
        for t_value, pedal_percent in self.get_corrected_samples():
            if t_value >= now - self.window_seconds:
                corrected_points.extend(self.point_to_canvas(t_value, pedal_percent, now))

        if len(corrected_points) >= 4:
            self.canvas.create_line(corrected_points, fill="#ef4444", width=3, smooth=True)

        points = []
        for t_value, pedal_percent in self.samples:
            if t_value >= now - self.window_seconds:
                points.extend(self.point_to_canvas(t_value, pedal_percent, now))

        if len(points) >= 4:
            self.canvas.create_line(points, fill="#38bdf8", width=4, smooth=True)

    def update(self):
        if not self.running:
            return

        elapsed = time.perf_counter() - self.start_time
        current_input = read_input_state(self.joystick_id)

        if current_input is None:
            pedal_percent = 0.0
            steering = 0
            self.brake = 0
        else:
            if self.previous_input is not None:
                previous_buttons = self.previous_input.dwButtons
                current_buttons = current_input.dwButtons

                mode_pressed = (
                    not is_button_down(previous_buttons, MODE_BUTTON)
                    and is_button_down(current_buttons, MODE_BUTTON)
                )
                emergency_pressed = (
                    not is_button_down(previous_buttons, EMERGENCY_BUTTON)
                    and is_button_down(current_buttons, EMERGENCY_BUTTON)
                )

                if mode_pressed:
                    self.mode = 1 if self.mode >= 4 else self.mode + 1
                    self.emergency = 0

                if emergency_pressed:
                    self.emergency = 0 if self.emergency else 1

            pedal_percent = round(y_to_accel_0_100(current_input.dwYpos), 1)
            steering = x_to_steering(current_input.dwXpos)
            self.brake = y_to_brake(current_input.dwYpos, self.brake)
            self.previous_input = current_input

        message = format_message(elapsed, pedal_percent, steering, self.brake, self.mode, self.emergency)
        self.sock.sendto(message.encode("utf-8"), self.address)

        if self.verbose:
            print(f"sent -> {self.ip}:{self.port}  {message}")

        self.samples.append((elapsed, pedal_percent))
        while self.samples and self.samples[0][0] < elapsed - self.window_seconds:
            self.samples.popleft()

        self.draw_dashboard(elapsed, pedal_percent, steering, self.brake, self.mode, self.emergency, current_input)

        self.root.after(self.interval_ms, self.update)

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read Windows wheel input and send it over UDP as "
            "'time,pedal_percent,steering,brake,mode,emergency'."
        )
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"Receiver IP. Default: {DEFAULT_IP}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Receiver UDP port. Default: {DEFAULT_PORT}")
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
        help=f"Send interval in seconds. Default: {DEFAULT_INTERVAL:g}",
    )
    parser.add_argument(
        "--graph-window",
        type=float,
        default=DEFAULT_GRAPH_WINDOW,
        help=f"Pedal graph time range in seconds. Default: {DEFAULT_GRAPH_WINDOW:g}",
    )
    parser.add_argument("--joystick-id", type=int, default=0, help="Windows joystick ID. Default: 0")
    parser.add_argument("--quiet", action="store_true", help="Send without printing every packet.")

    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0.")
    if args.graph_window <= 0:
        raise ValueError("--graph-window must be greater than 0.")

    LivePedalUdpPlot(
        args.ip,
        args.port,
        args.interval,
        args.joystick_id,
        not args.quiet,
        args.corrected_port,
        args.graph_window,
    ).run()


if __name__ == "__main__":
    main()
