# Final Jetson runtime command example:
# MPLCONFIGDIR=/tmp/matplotlib python3 jetson_main.py --serial-port /dev/ttyUSB0 --video-udp-host <NOTEBOOK_IP>

import argparse
import json
import socket
import threading
import time
from collections import deque
import csv
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent


# Integrated from src/model_runtime.py so this final script has no local Python module dependency.
def load_runtime_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def build_transform(config: dict) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config["image_height"], config["image_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ]
    )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(config_path: Path, device_arg: str | None = None) -> tuple[nn.Module, dict, torch.device]:
    config = load_runtime_config(config_path)
    config_dir = config_path.parent.parent
    weights_path = config_dir / config["weights_path"]
    device = resolve_device(device_arg)

    model = build_model(config["model_name"], len(config["class_names"]))
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, device


@torch.inference_mode()
def predict_image(model: nn.Module, config: dict, device: torch.device, image) -> dict[str, float | str]:
    transform = build_transform(config)
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred_idx = int(torch.argmax(probs).item())
    prediction = config["class_names"][pred_idx]
    e_mapping = config.get("e_mapping", {})

    off_score = float(probs[0].item())
    on_score = float(probs[1].item())

    return {
        "prediction": prediction,
        "e_raw": int(e_mapping.get(prediction, pred_idx)),
        "confidence": float(probs[pred_idx].item()),
        "off_score": off_score,
        "on_score": on_score,
    }

try:
    import serial
except ImportError:
    serial = None


STATE_NAME = "pedal_baseline_state.json"
INITIAL_STATE_NAME = "pedal_baseline_state_initial.json"
WARMUP_STATE_NAME = "pedal_baseline_state_warmup.json"
DEFAULT_STATE_PATH = ROOT_DIR / STATE_NAME
DEFAULT_INITIAL_STATE_PATH = ROOT_DIR / INITIAL_STATE_NAME
DEFAULT_WARMUP_STATE_PATH = ROOT_DIR / WARMUP_STATE_NAME
DEFAULT_UDP_IP = "0.0.0.0"
DEFAULT_UDP_PORT = 5005
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_BAUDRATE = 9600
DEFAULT_LOG_CSV = "outputs/jetson_runtime_log.csv"
DEFAULT_TRANSITION_CSV = "outputs/baseline_transition_history.csv"
DEFAULT_PLOT_WINDOW_SEC = 30.0
DEFAULT_PLOT_VIDEO = "outputs/jetson_runtime_plot.mp4"
DEFAULT_ENV_CONFIG = ROOT_DIR / "config" / "runtime_config.json"
DEFAULT_SAMPLES_DIR = ROOT_DIR / "samples"
DEFAULT_MODE_IMAGE_DIR = Path(__file__).resolve().parent / "mode_jpg"
DEFAULT_CAMERA_DEVICE = "/dev/video0"
DEFAULT_VIDEO_WIDTH = 960
DEFAULT_VIDEO_HEIGHT = 540
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_UDP_PORT = 5000
DEFAULT_TELEMETRY_UDP_PORT = 5006
DEFAULT_INFER_EVERY = 1
DEFAULT_ENV_WARMUP = 2
VIDEO_WINDOW_NAME = "Jetson Control Video Stream"

PEDAL_THRESHOLD = 5.0
HIGH_PEDAL_THRESHOLD = 80.0
HIGH_PEDAL_HOLD_SEC = 2.0
SLOPE_STEP_COUNT = 4
ENV_SWITCH_THRESHOLD = 50
NORMAL_UPDATE_Z = 2.0
OUTLIER_Z = 3.0
WARN_UPDATE_WEIGHT = 0.5
MIN_SIGMA = 0.1
TRANSITION_SAMPLE_COUNT = 60.0
ENVIRONMENT_ENABLE_DEFAULT = 1.0
FORCE_ENVIRONMENT_ENABLE_ON = False


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def empty_baseline_state():
    return {
        "sample_count": 0.0,
        "mean": 0.0,
        "m2": 0.0,
        "variance": 0.0,
        "sigma": MIN_SIGMA,
        "learned_event_count": 0,
    }


def refresh_state_stats(state):
    n = float(state["sample_count"])
    mean = float(state["mean"])
    m2 = float(state.get("m2", 0.0))

    saved_variance = float(state.get("variance", 0.0))
    if n <= 0:
        variance = 0.0
        sigma = MIN_SIGMA
    elif n <= 1:
        if saved_variance > 0:
            variance = max(saved_variance, MIN_SIGMA * MIN_SIGMA)
            sigma = variance ** 0.5
        else:
            sigma = max(abs(mean) * 0.05, MIN_SIGMA)
            variance = sigma * sigma
    else:
        variance = max(m2 / (n - 1.0), MIN_SIGMA * MIN_SIGMA)
        sigma = variance ** 0.5

    state["sample_count"] = n
    state["mean"] = mean
    state["m2"] = m2
    state["variance"] = float(variance)
    state["sigma"] = float(sigma)
    state["learned_event_count"] = int(state.get("learned_event_count", 0))
    return state


def update_baseline_state(state, value, weight=1.0):
    weight = float(weight)
    if weight <= 0:
        return refresh_state_stats(state)

    value = float(value)
    n = float(state["sample_count"])
    mean = float(state["mean"])
    m2 = float(state.get("m2", 0.0))
    new_n = n + weight

    if n <= 0:
        new_mean = value
        new_m2 = 0.0
    else:
        delta = value - mean
        new_mean = mean + (weight / new_n) * delta
        new_m2 = m2 + weight * delta * (value - new_mean)

    state["sample_count"] = new_n
    state["mean"] = new_mean
    state["m2"] = new_m2
    state["learned_event_count"] = int(state.get("learned_event_count", 0)) + 1
    return refresh_state_stats(state)


def state_ready(state):
    return float(state["sample_count"]) > 0


def load_baseline_state(state_path):
    if not state_path.exists():
        return None
    if state_path.stat().st_size == 0:
        return None

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    required = {"sample_count", "mean", "m2"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"{state_path} is missing saved fields: {sorted(missing)}")

    return refresh_state_stats(state)


def load_or_empty_state(state_path):
    state = load_baseline_state(Path(state_path))
    return state if state is not None else empty_baseline_state()


def save_baseline_state(state_path, state):
    state = refresh_state_stats(dict(state))
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state


def state_from_mean_variance(mean, variance, sample_count=0.0, learned_event_count=0):
    sample_count = float(sample_count)
    variance = max(float(variance), MIN_SIGMA * MIN_SIGMA)
    if sample_count > 1:
        m2 = variance * (sample_count - 1.0)
    else:
        m2 = 0.0
    return {
        "sample_count": sample_count,
        "mean": float(mean),
        "m2": float(m2),
        "variance": variance,
        "sigma": variance ** 0.5,
        "learned_event_count": int(learned_event_count),
    }


def z_score(value, mean, sigma):
    return (value - mean) / sigma


def classify_z(value, mean, sigma):
    z = abs(z_score(value, mean, sigma))
    if z >= OUTLIER_Z:
        return "OUTLIER"
    if z >= NORMAL_UPDATE_Z:
        return "WARN"
    return "OK"


def baseline_update_weight(value, mean, sigma):
    z = abs(z_score(value, mean, sigma))
    if z < NORMAL_UPDATE_Z:
        return 1.0, "learned"
    if z < OUTLIER_Z:
        return WARN_UPDATE_WEIGHT, f"weak learned ({WARN_UPDATE_WEIGHT:g}x)"
    return 0.0, "held out"


def normal_pdf(x, mean, sigma):
    sigma = max(float(sigma), MIN_SIGMA)
    return np.exp(-0.5 * ((x - float(mean)) / sigma) ** 2) / (sigma * (2.0 * np.pi) ** 0.5)


def js_divergence_normalized(state_a, state_b, points=512):
    mean_a = float(state_a["mean"])
    sigma_a = max(float(state_a["sigma"]), MIN_SIGMA)
    mean_b = float(state_b["mean"])
    sigma_b = max(float(state_b["sigma"]), MIN_SIGMA)

    lower = min(mean_a - 6.0 * sigma_a, mean_b - 6.0 * sigma_b)
    upper = max(mean_a + 6.0 * sigma_a, mean_b + 6.0 * sigma_b)
    if lower == upper:
        return 0.0

    x = np.linspace(lower, upper, int(points))
    p = normal_pdf(x, mean_a, sigma_a)
    q = normal_pdf(x, mean_b, sigma_b)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.trapz(p * np.log((p + eps) / (m + eps)), x)
    kl_qm = np.trapz(q * np.log((q + eps) / (m + eps)), x)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return clamp(float(js / np.log(2.0)), 0.0, 1.0)


def transition_alpha(initial_state, warmup_state, transition_samples=TRANSITION_SAMPLE_COUNT):
    n = float(warmup_state["sample_count"])
    if n <= 0 or not state_ready(warmup_state):
        return 0.0, 0.0, 1.0, 0.0

    r = clamp(n / float(transition_samples), 0.0, 1.0)
    if r >= 1.0:
        return 1.0, r, 1.0, 0.0

    comparison_warmup = dict(warmup_state)
    sigma_floor = max(float(initial_state["sigma"]) * (1.0 - r), MIN_SIGMA)
    if float(comparison_warmup["sigma"]) < sigma_floor:
        comparison_warmup["sigma"] = sigma_floor
        comparison_warmup["variance"] = sigma_floor * sigma_floor
    d = js_divergence_normalized(initial_state, comparison_warmup)
    j = clamp(1.0 - d * (1.0 - r), 0.0, 1.0)
    alpha = clamp(r * j, 0.0, 1.0)
    return alpha, r, j, d


def blend_baseline_states(initial_state, warmup_state, transition_samples=TRANSITION_SAMPLE_COUNT):
    if not state_ready(initial_state):
        return refresh_state_stats(dict(warmup_state)), 1.0, 1.0, 1.0, 0.0
    if not state_ready(warmup_state):
        return refresh_state_stats(dict(initial_state)), 0.0, 0.0, 1.0, 0.0

    alpha, r, j, d = transition_alpha(initial_state, warmup_state, transition_samples)
    if alpha >= 1.0:
        return refresh_state_stats(dict(warmup_state)), alpha, r, j, d

    mean = (1.0 - alpha) * float(initial_state["mean"]) + alpha * float(warmup_state["mean"])
    variance = (1.0 - alpha) * float(initial_state["variance"]) + alpha * float(warmup_state["variance"])
    sample_count = float(warmup_state["sample_count"])
    learned_event_count = int(warmup_state.get("learned_event_count", 0))
    return state_from_mean_variance(mean, variance, sample_count, learned_event_count), alpha, r, j, d


def state_history_fields(prefix):
    return [
        f"{prefix}_sample_count",
        f"{prefix}_mean",
        f"{prefix}_variance",
        f"{prefix}_sigma",
        f"{prefix}_learned_event_count",
    ]


def state_history_values(prefix, state):
    state = refresh_state_stats(dict(state))
    return {
        f"{prefix}_sample_count": f"{state['sample_count']:.6f}",
        f"{prefix}_mean": f"{state['mean']:.6f}",
        f"{prefix}_variance": f"{state['variance']:.6f}",
        f"{prefix}_sigma": f"{state['sigma']:.6f}",
        f"{prefix}_learned_event_count": int(state.get("learned_event_count", 0)),
    }


def init_transition_history_csv(path):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wall_time",
        "event_end_time_sec",
        "event_slope",
        "learn_weight",
        "action",
        "transition_alpha",
        "transition_r",
        "transition_j",
        "transition_d",
    ]
    fieldnames.extend(state_history_fields("a_initial"))
    fieldnames.extend(state_history_fields("b_warmup"))
    fieldnames.extend(state_history_fields("c_active"))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    return path


def append_transition_history_csv(path, event_end_time_sec, event_slope, learn_weight, action, controller):
    if path is None:
        return
    fieldnames = [
        "wall_time",
        "event_end_time_sec",
        "event_slope",
        "learn_weight",
        "action",
        "transition_alpha",
        "transition_r",
        "transition_j",
        "transition_d",
    ]
    fieldnames.extend(state_history_fields("a_initial"))
    fieldnames.extend(state_history_fields("b_warmup"))
    fieldnames.extend(state_history_fields("c_active"))
    row = {
        "wall_time": f"{time.time():.3f}",
        "event_end_time_sec": f"{event_end_time_sec:.6f}",
        "event_slope": "" if event_slope is None else f"{event_slope:.6f}",
        "learn_weight": f"{learn_weight:.6f}",
        "action": action,
        "transition_alpha": f"{controller.transition_alpha:.6f}",
        "transition_r": f"{controller.transition_r:.6f}",
        "transition_j": f"{controller.transition_j:.6f}",
        "transition_d": f"{controller.transition_d:.6f}",
    }
    row.update(state_history_values("a_initial", controller.initial_state))
    row.update(state_history_values("b_warmup", controller.warmup_state))
    row.update(state_history_values("c_active", controller.state))
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def control_gain_from_z(z):
    z_abs = abs(float(z))
    if z_abs < NORMAL_UPDATE_Z:
        return 0.0
    if z_abs >= OUTLIER_Z:
        return 1.0

    ratio = (z_abs - NORMAL_UPDATE_Z) / (OUTLIER_Z - NORMAL_UPDATE_Z)
    return ratio ** 3


def control_lambda(reference_slope, current_slope, z, environment_enable):
    if current_slope is None or current_slope <= 0:
        return 1.0, 0.0
    if reference_slope is None or reference_slope <= 0:
        return 1.0, 0.0

    ratio = clamp(reference_slope / current_slope, 0.0, 1.0)
    k = control_gain_from_z(z)
    lam = 1.0 - environment_enable * k * (1.0 - ratio)
    return clamp(lam, 0.0, 1.0), k


def representative_slope(samples, step_count=SLOPE_STEP_COUNT):
    if len(samples) < step_count + 1:
        return None

    best_slope = None
    data = list(samples)
    for end_idx in range(step_count, len(data)):
        window = data[end_idx - step_count : end_idx + 1]
        slopes = []
        for (t0, p0), (t1, p1) in zip(window, window[1:]):
            dt = t1 - t0
            if dt > 0:
                slopes.append((p1 - p0) / dt)

        if not slopes:
            continue

        rolling_slope = sum(slopes) / len(slopes)
        if best_slope is None or rolling_slope > best_slope:
            best_slope = rolling_slope

    return best_slope


def parse_udp_message(message):
    parts = message.strip().split(",")
    if len(parts) not in {2, 6}:
        raise ValueError(
            "Expected 'time_sec,pedal_percent' or "
            f"'time_sec,pedal_percent,steering,brake,mode,emergency', got: {message!r}"
        )
    mode = int(float(parts[4])) if len(parts) == 6 else 1
    if mode not in {1, 2, 3, 4}:
        raise ValueError(f"Expected mode 1, 2, 3, or 4, got: {mode!r}")
    emergency = int(float(parts[5])) if len(parts) == 6 else 0
    emergency = 1 if emergency == 1 else 0
    return float(parts[0]), float(parts[1]), mode, emergency


def pedal_percent_to_pwm(pedal_percent):
    pedal_percent = clamp(float(pedal_percent), 0.0, 100.0)
    return round(150 + (pedal_percent / 100.0) * (255 - 150))


def send_line(ser, text):
    ser.write((text + "\n").encode("utf-8"))


def motor_command_from_pedal(corrected_pedal):
    corrected_pedal = clamp(corrected_pedal, 0.0, 100.0)
    pwm = pedal_percent_to_pwm(corrected_pedal)
    if pwm <= 150:
        return "STOP"
    return str(pwm)


def motor_command_with_environment(corrected_pedal, environment_enable):
    command = motor_command_from_pedal(corrected_pedal)
    e_value = 1 if float(environment_enable) >= 0.5 else 0
    return f"{command},{e_value}"


def send_motor_command(ser, corrected_pedal, environment_enable):
    command = motor_command_with_environment(corrected_pedal, environment_enable)
    send_line(ser, command)
    return command


class SampleEnvironmentClassifier:
    def __init__(self, config_path, samples_dir, device_arg=None, switch_threshold=ENV_SWITCH_THRESHOLD):
        self.config_path = Path(config_path).resolve()
        self.samples_dir = Path(samples_dir).resolve()
        self.model, self.config, self.device = load_classifier(self.config_path, device_arg)
        self.image_paths = sorted(
            [p for p in self.samples_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        )
        if not self.image_paths:
            raise RuntimeError(f"No sample images found in {self.samples_dir}")
        self.index = 0
        self._lock = threading.Lock()
        self.switch_threshold = int(switch_threshold)
        self.effective_prediction = "off"
        self.pending_prediction = None
        self.pending_count = 0
        self.last_result = {
            "raw_prediction": "off",
            "prediction": "off",
            "e_value": 0.0,
            "image_name": self.image_paths[0].name,
            "inference_ms": 0.0,
            "fps": 0.0,
            "pending_prediction": "",
            "pending_count": 0,
        }

    def step(self):
        with self._lock:
            image_path = self.image_paths[self.index]
            self.index = (self.index + 1) % len(self.image_paths)

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = predict_image(self.model, self.config, self.device, image)
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000.0
        raw_prediction = str(result["prediction"]).lower()
        with self._lock:
            if raw_prediction == self.effective_prediction:
                self.pending_prediction = None
                self.pending_count = 0
            else:
                if raw_prediction == self.pending_prediction:
                    self.pending_count += 1
                else:
                    self.pending_prediction = raw_prediction
                    self.pending_count = 1

                if self.pending_count >= self.switch_threshold:
                    self.effective_prediction = raw_prediction
                    self.pending_prediction = None
                    self.pending_count = 0

            prediction = self.effective_prediction
            pending_prediction = self.pending_prediction or ""
            pending_count = self.pending_count
        e_value = 1.0 if prediction == "on" else 0.0
        fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

        result_snapshot = {
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "e_value": e_value,
            "image_name": image_path.name,
            "inference_ms": inference_ms,
            "fps": fps,
            "pending_prediction": pending_prediction,
            "pending_count": pending_count,
        }
        with self._lock:
            self.last_result = result_snapshot
            return dict(self.last_result)

    def current(self):
        with self._lock:
            return dict(self.last_result)


class BackgroundEnvironmentMonitor:
    def __init__(self, classifier):
        self.classifier = classifier
        self.stop_event = threading.Event()
        self.thread = None
        self.last_error = None

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.classifier.step()
            except Exception as exc:
                self.last_error = exc
                self.stop_event.set()
                break

    def start(self):
        self.thread = threading.Thread(target=self._run, name="environment-monitor", daemon=True)
        self.thread.start()

    def current(self):
        if self.last_error is not None:
            raise RuntimeError(f"Environment monitor stopped: {self.last_error}") from self.last_error
        return self.classifier.current()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class VideoOverlayState:
    def __init__(self, initial_state=0):
        self._lock = threading.Lock()
        self._state = int(initial_state)

    def set(self, state):
        with self._lock:
            self._state = 1 if int(state) == 1 else 0

    def get(self):
        with self._lock:
            return self._state


class ModeState:
    def __init__(self, initial_mode=1):
        self._lock = threading.Lock()
        self._mode = int(initial_mode)

    def set(self, mode):
        mode = int(mode)
        if mode not in {1, 2, 3, 4}:
            mode = 1
        with self._lock:
            self._mode = mode

    def get(self):
        with self._lock:
            return self._mode


class EmergencyOverlayState:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False

    def set(self, active):
        with self._lock:
            self._active = bool(active)

    def get(self):
        with self._lock:
            return self._active


class RealtimeEnvironmentState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "raw_prediction": "off",
            "prediction": "off",
            "e_value": 0.0,
            "image_name": "camera",
            "inference_ms": 0.0,
            "fps": 0.0,
            "pending_prediction": "",
            "pending_count": 0,
        }

    def update(self, raw_prediction, prediction, inference_ms, fps, pending_prediction="", pending_count=0):
        e_value = 1.0 if str(prediction).lower() == "on" else 0.0
        with self._lock:
            self._state = {
                "raw_prediction": str(raw_prediction).lower(),
                "prediction": str(prediction).lower(),
                "e_value": e_value,
                "image_name": "camera",
                "inference_ms": float(inference_ms),
                "fps": float(fps),
                "pending_prediction": pending_prediction or "",
                "pending_count": int(pending_count),
            }

    def current(self):
        with self._lock:
            return dict(self._state)


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


def open_video_udp_stream(host, port, width, height, fps):
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
    import torch

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


def draw_status_overlay(frame, state, emergency_active=False):
    label = "ON" if state == 1 else "OFF"
    color = (0, 220, 0) if state == 1 else (0, 0, 255)
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.6, height / 190.0)
    thickness = max(3, int(height / 150))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = (width - text_w) // 2
    y = height - max(28, int(height * 0.055))

    if emergency_active and state == 0:
        pad = max(16, int(height * 0.025))
        cv2.rectangle(
            frame,
            (max(0, x - pad), max(0, y - text_h - pad)),
            (min(width - 1, x + text_w + pad), min(height - 1, y + baseline + pad)),
            (0, 210, 255),
            max(3, int(height / 120)),
        )
        cv2.putText(frame, label, (x + 3, y + 3), font, font_scale, (0, 210, 255), thickness + 5, cv2.LINE_AA)
    cv2.putText(frame, label, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_video_preview_status(frame, device, fps_value):
    text = f"{device} | control video | {fps_value:4.1f} FPS | q/ESC quit"
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


class VideoStreamer:
    def __init__(
        self,
        device,
        width,
        height,
        fps,
        udp_host,
        udp_port,
        overlay_state,
        no_display=True,
        windowed=False,
        window_width=960,
        window_height=540,
        flip=False,
        environment_state=None,
        model=None,
        config=None,
        model_device=None,
        infer_every=DEFAULT_INFER_EVERY,
        switch_threshold=ENV_SWITCH_THRESHOLD,
        update_overlay_from_inference=True,
        mode_state=None,
        mode_image_dir=DEFAULT_MODE_IMAGE_DIR,
        emergency_overlay_state=None,
    ):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.udp_host = udp_host
        self.udp_port = int(udp_port)
        self.overlay_state = overlay_state
        self.no_display = bool(no_display)
        self.windowed = bool(windowed)
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self.flip = bool(flip)
        self.environment_state = environment_state
        self.model = model
        self.config = config
        self.model_device = model_device
        self.infer_every = int(infer_every)
        self.switch_threshold = int(switch_threshold)
        self.update_overlay_from_inference = bool(update_overlay_from_inference)
        self.mode_state = mode_state
        self.mode_image_dir = Path(mode_image_dir)
        self.emergency_overlay_state = emergency_overlay_state
        self.stop_event = threading.Event()
        self.thread = None
        self.last_error = None

    def load_mode_frames(self):
        frames = {}
        for mode in (2, 3, 4):
            image_path = self.mode_image_dir / f"{mode}.jpg"
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read mode image: {image_path}")
            frames[mode] = resize_to_16_9(frame, self.width, self.height)
        return frames

    def start(self):
        self.thread = threading.Thread(target=self._run, name="video-streamer", daemon=True)
        self.thread.start()

    def _run(self):
        cap = None
        udp_writer = None
        try:
            cap = open_camera(self.device, self.width, self.height, self.fps)
            udp_writer = open_video_udp_stream(self.udp_host, self.udp_port, self.width, self.height, self.fps)
            mode_frames = self.load_mode_frames()
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video camera opened {self.device}: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
            if udp_writer is not None:
                print(f"Video streaming to udp://{self.udp_host}:{self.udp_port}")

            preview_width = self.window_width if self.windowed else self.width
            preview_height = self.window_height if self.windowed else self.height
            if not self.no_display:
                cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
                if self.windowed:
                    cv2.resizeWindow(VIDEO_WINDOW_NAME, self.window_width, self.window_height)
                else:
                    cv2.setWindowProperty(VIDEO_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            frames = 0
            frame_count = 0
            fps_value = 0.0
            inference_ms = 0.0
            effective_state = None
            pending_state = None
            pending_count = 0
            last_time = time.monotonic()

            while not self.stop_event.is_set():
                loop_started_at = time.monotonic()
                mode = self.mode_state.get() if self.mode_state is not None else 1
                if mode == 1:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Failed to read frame from camera.")

                    if self.flip:
                        frame = cv2.flip(frame, 1)
                    stream_frame = resize_to_16_9(frame, self.width, self.height)
                else:
                    stream_frame = mode_frames[mode].copy()

                frame_count += 1

                if (
                    self.environment_state is not None
                    and self.model is not None
                    and self.config is not None
                    and self.model_device is not None
                    and frame_count % self.infer_every == 0
                ):
                    image = frame_to_pil_rgb(stream_frame)
                    if self.model_device.type == "cuda":
                        import torch
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = predict_image(self.model, self.config, self.model_device, image)
                    if self.model_device.type == "cuda":
                        import torch
                        torch.cuda.synchronize()
                    inference_ms = (time.perf_counter() - start) * 1000.0

                    raw_prediction = str(result["prediction"]).lower()
                    effective_state, pending_state, pending_count = update_effective_state(
                        raw_prediction,
                        effective_state,
                        pending_state,
                        pending_count,
                        self.switch_threshold,
                    )
                    self.environment_state.update(
                        raw_prediction,
                        effective_state,
                        inference_ms,
                        1000.0 / inference_ms if inference_ms > 0 else 0.0,
                        pending_state or "",
                        pending_count,
                    )
                    if self.update_overlay_from_inference:
                        self.overlay_state.set(1 if effective_state == "on" else 0)

                draw_status_overlay(
                    stream_frame,
                    self.overlay_state.get(),
                    self.emergency_overlay_state.get() if self.emergency_overlay_state is not None else False,
                )

                if udp_writer is not None:
                    udp_writer.write(stream_frame)

                frames += 1
                now = time.monotonic()
                elapsed = now - last_time
                if elapsed >= 1.0:
                    fps_value = frames / elapsed
                    frames = 0
                    last_time = now

                if mode != 1 and self.fps > 0:
                    sleep_time = (1.0 / self.fps) - (time.monotonic() - loop_started_at)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if self.no_display:
                    continue

                display_frame = stream_frame
                if preview_width != self.width or preview_height != self.height:
                    display_frame = cv2.resize(stream_frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
                draw_video_preview_status(display_frame, self.device, fps_value)
                cv2.imshow(VIDEO_WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.stop_event.set()
                    break
        except Exception as exc:
            self.last_error = exc
            print(f"Video streamer stopped with error: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if udp_writer is not None:
                udp_writer.release()
            if not self.no_display:
                cv2.destroyWindow(VIDEO_WINDOW_NAME)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class PedalController:
    def __init__(
        self,
        state_path,
        initial_state_path=DEFAULT_INITIAL_STATE_PATH,
        warmup_state_path=DEFAULT_WARMUP_STATE_PATH,
        transition_samples=TRANSITION_SAMPLE_COUNT,
        reset_warmup_on_start=True,
        transition_history_path=DEFAULT_TRANSITION_CSV,
    ):
        self.state_path = Path(state_path)
        self.initial_state_path = Path(initial_state_path)
        self.warmup_state_path = Path(warmup_state_path)
        self.transition_history_path = init_transition_history_csv(transition_history_path)
        self.transition_samples = float(transition_samples)
        saved_initial = load_baseline_state(self.initial_state_path)
        if saved_initial is None:
            saved_initial = load_baseline_state(self.state_path)
        self.initial_state = saved_initial if saved_initial is not None else empty_baseline_state()
        self.warmup_state = empty_baseline_state() if reset_warmup_on_start else load_or_empty_state(self.warmup_state_path)
        self.state, self.transition_alpha, self.transition_r, self.transition_j, self.transition_d = blend_baseline_states(
            self.initial_state, self.warmup_state, self.transition_samples
        )
        save_baseline_state(self.warmup_state_path, self.warmup_state)
        save_baseline_state(self.state_path, self.state)
        self.current_event = []
        self.in_event = False
        self.event_environment_enable = 0.0
        self.event_environment_prediction = "off"
        self.event_environment_image = "none"
        self.event_environment_inference_ms = 0.0
        self.event_environment_fps = 0.0
        self.high_pedal_started_at = None
        self.block_until_event_end = False

    def refresh_active_state(self):
        self.state, self.transition_alpha, self.transition_r, self.transition_j, self.transition_d = blend_baseline_states(
            self.initial_state, self.warmup_state, self.transition_samples
        )
        save_baseline_state(self.state_path, self.state)
        return self.state

    def commit_transition_state(self):
        n = float(self.warmup_state["sample_count"])
        if n <= 0:
            next_initial = self.initial_state
        elif n < self.transition_samples:
            next_initial = self.state
        else:
            next_initial = self.warmup_state

        self.initial_state = save_baseline_state(self.initial_state_path, next_initial)
        self.state = save_baseline_state(self.state_path, self.initial_state)
        return self.initial_state

    def finalize_drive(self):
        if self.in_event:
            self.finish_event()
            self.in_event = False
            self.current_event = []
        self.commit_transition_state()

    def process_sample(self, time_sec, raw_pedal, env_snapshot, emergency=0):
        raw_pedal = clamp(raw_pedal, 0.0, 100.0)
        emergency = 1 if int(emergency) == 1 else 0
        corrected_pedal = raw_pedal
        current_slope = None
        z = None
        label = "OK"
        lam = 1.0
        k = 0.0
        action = "streaming"
        current_environment_enable = 1.0 if FORCE_ENVIRONMENT_ENABLE_ON else float(env_snapshot["e_value"])
        current_environment_prediction = "forced_on" if FORCE_ENVIRONMENT_ENABLE_ON else str(env_snapshot["prediction"])
        current_environment_image = str(env_snapshot["image_name"])
        current_environment_inference_ms = float(env_snapshot["inference_ms"])
        current_environment_fps = float(env_snapshot["fps"])
        emergency_active = emergency == 1 and current_environment_enable < 0.5
        arduino_environment_enable = 1.0 if emergency_active else current_environment_enable
        environment_enable = self.event_environment_enable
        environment_prediction = self.event_environment_prediction
        environment_image = self.event_environment_image
        environment_inference_ms = self.event_environment_inference_ms
        environment_fps = self.event_environment_fps

        if raw_pedal >= PEDAL_THRESHOLD:
            if not self.in_event:
                self.current_event = []
                self.in_event = True
                self.event_environment_enable = current_environment_enable
                self.event_environment_prediction = current_environment_prediction
                self.event_environment_image = current_environment_image
                self.event_environment_inference_ms = current_environment_inference_ms
                self.event_environment_fps = current_environment_fps
                self.high_pedal_started_at = None
                self.block_until_event_end = False
                environment_enable = self.event_environment_enable
                environment_prediction = self.event_environment_prediction
                environment_image = self.event_environment_image
                environment_inference_ms = self.event_environment_inference_ms
                environment_fps = self.event_environment_fps

            self.current_event.append((time_sec, raw_pedal))

            if environment_enable >= 0.5:
                if state_ready(self.state):
                    current_slope = representative_slope(self.current_event)
                    if current_slope is not None:
                        z = z_score(current_slope, self.state["mean"], self.state["sigma"])
                        label = classify_z(current_slope, self.state["mean"], self.state["sigma"])
                        lam, k = control_lambda(self.state["mean"], current_slope, z, environment_enable)
                        corrected_pedal = raw_pedal * lam

                        # Condition 2: only for fast-side WARN/OUTLIER in an ON environment.
                        is_fast_outlier = z is not None and z >= NORMAL_UPDATE_Z
                        if label in {"WARN", "OUTLIER"} and is_fast_outlier:
                            if raw_pedal >= HIGH_PEDAL_THRESHOLD:
                                if self.high_pedal_started_at is None:
                                    self.high_pedal_started_at = time_sec
                                elif (time_sec - self.high_pedal_started_at) >= HIGH_PEDAL_HOLD_SEC:
                                    self.block_until_event_end = True
                            else:
                                self.high_pedal_started_at = None
                        else:
                            self.high_pedal_started_at = None
                    else:
                        self.high_pedal_started_at = None
                else:
                    self.high_pedal_started_at = None
            else:
                self.high_pedal_started_at = None
                if emergency_active and raw_pedal >= HIGH_PEDAL_THRESHOLD:
                    self.block_until_event_end = True

            if self.block_until_event_end:
                corrected_pedal = 0.0
                action = "emergency_blocked_until_event_end" if emergency_active else "blocked_until_event_end"
        else:
            if self.in_event:
                action = self.finish_event()
            self.in_event = False
            self.current_event = []
            self.event_environment_enable = 0.0
            self.event_environment_prediction = "off"
            self.event_environment_image = "none"
            self.event_environment_inference_ms = 0.0
            self.event_environment_fps = 0.0
            self.high_pedal_started_at = None
            self.block_until_event_end = False

        return {
            "time_sec": time_sec,
            "raw_pedal": raw_pedal,
            "corrected_pedal": corrected_pedal,
            "slope": current_slope,
            "z": z,
            "label": label,
            "lambda": lam,
            "k": k,
            "action": action,
            "state": refresh_state_stats(dict(self.state)),
            "warmup_state": refresh_state_stats(dict(self.warmup_state)),
            "transition_alpha": self.transition_alpha,
            "transition_r": self.transition_r,
            "transition_j": self.transition_j,
            "transition_d": self.transition_d,
            "current_environment_enable": current_environment_enable,
            "current_environment_prediction": current_environment_prediction,
            "arduino_environment_enable": arduino_environment_enable,
            "emergency": emergency,
            "emergency_active": emergency_active,
            "environment_enable": environment_enable,
            "environment_prediction": environment_prediction,
            "environment_image": environment_image,
            "environment_inference_ms": environment_inference_ms,
            "environment_fps": environment_fps,
            "blocked_until_event_end": self.block_until_event_end,
        }

    def finish_event(self):
        if not self.current_event:
            return "empty event"

        slope = representative_slope(self.current_event)
        if slope is None:
            return "short event"

        if not state_ready(self.state):
            return "not judged"

        weight, action = baseline_update_weight(slope, self.state["mean"], self.state["sigma"])
        if weight > 0:
            update_baseline_state(self.warmup_state, slope, weight)
            save_baseline_state(self.warmup_state_path, self.warmup_state)
        self.refresh_active_state()
        if float(self.warmup_state["sample_count"]) >= self.transition_samples:
            save_baseline_state(self.initial_state_path, self.warmup_state)
            self.initial_state = refresh_state_stats(dict(self.warmup_state))
        append_transition_history_csv(
            self.transition_history_path,
            self.current_event[-1][0],
            slope,
            weight,
            action,
            self,
        )
        return action


def open_udp_socket(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    return sock


def open_telemetry_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_corrected_pedal_telemetry(sock, host, port, time_sec, corrected_pedal):
    if sock is None or not host:
        return
    text = f"{float(time_sec):.6f},{float(corrected_pedal):.3f}"
    sock.sendto(text.encode("utf-8"), (host, int(port)))


def open_serial(port, baudrate):
    if serial is None:
        raise RuntimeError("pyserial is not installed. Install it with: pip install pyserial")
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2.0)
    return ser


def init_log_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recv_wall_time",
                "src_ip",
                "src_port",
                "time_sec",
                "raw_pedal",
                "corrected_pedal",
                "motor_command",
                "label",
                "z",
                "slope",
                "k",
                "lambda",
                "environment_enable",
                "environment_prediction",
                "environment_image",
                "environment_inference_ms",
                "environment_fps",
                "blocked_until_event_end",
                "sample_count",
                "warmup_sample_count",
                "transition_alpha",
                "transition_r",
                "transition_j",
                "transition_d",
                "action",
            ],
        )
        writer.writeheader()
    return path


def append_log_csv(path, address, result, motor_command):
    path = Path(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recv_wall_time",
                "src_ip",
                "src_port",
                "time_sec",
                "raw_pedal",
                "corrected_pedal",
                "motor_command",
                "label",
                "z",
                "slope",
                "k",
                "lambda",
                "environment_enable",
                "environment_prediction",
                "environment_image",
                "environment_inference_ms",
                "environment_fps",
                "blocked_until_event_end",
                "sample_count",
                "warmup_sample_count",
                "transition_alpha",
                "transition_r",
                "transition_j",
                "transition_d",
                "action",
            ],
        )
        writer.writerow(
            {
                "recv_wall_time": f"{time.time():.3f}",
                "src_ip": address[0],
                "src_port": address[1],
                "time_sec": f"{result['time_sec']:.3f}",
                "raw_pedal": f"{result['raw_pedal']:.3f}",
                "corrected_pedal": f"{result['corrected_pedal']:.3f}",
                "motor_command": motor_command,
                "label": result["label"],
                "z": "" if result["z"] is None else f"{result['z']:.6f}",
                "slope": "" if result["slope"] is None else f"{result['slope']:.6f}",
                "k": f"{result['k']:.6f}",
                "lambda": f"{result['lambda']:.6f}",
                "environment_enable": f"{result['environment_enable']:.1f}",
                "environment_prediction": result["environment_prediction"],
                "environment_image": result["environment_image"],
                "environment_inference_ms": f"{result['environment_inference_ms']:.3f}",
                "environment_fps": f"{result['environment_fps']:.3f}",
                "blocked_until_event_end": int(bool(result["blocked_until_event_end"])),
                "sample_count": f"{result['state']['sample_count']:.3f}",
                "warmup_sample_count": f"{result['warmup_state']['sample_count']:.3f}",
                "transition_alpha": f"{result['transition_alpha']:.6f}",
                "transition_r": f"{result['transition_r']:.6f}",
                "transition_j": f"{result['transition_j']:.6f}",
                "transition_d": f"{result['transition_d']:.6f}",
                "action": result["action"],
            }
        )


class LivePlotter:
    def __init__(self, window_sec=DEFAULT_PLOT_WINDOW_SEC, record_path=None, record_fps=20.0):
        self.window_sec = float(window_sec)
        self.raw_samples = deque()
        self.corrected_samples = deque()
        self.stop_requested = False
        self.record_path = Path(record_path).resolve() if record_path else None
        self.record_fps = float(record_fps)
        self.writer = None

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(11, 5))
        (self.raw_line,) = self.ax.plot([], [], color="#1f77b4", linewidth=2, label="raw pedal")
        (self.corrected_line,) = self.ax.plot([], [], color="#d62728", linewidth=2, label="corrected pedal")
        self.ax.set_title("Jetson Real-time Pedal Control")
        self.ax.set_xlabel("time (sec)")
        self.ax.set_ylabel("pedal (%)")
        self.ax.set_ylim(-5, 105)
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper left")
        self.ax.set_facecolor("#ffe8e8")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        if event.key in {"q", "escape"}:
            self.stop_requested = True

    def update(self, time_sec, raw_pedal, corrected_pedal, label, lambda_value, environment_enable, env_prediction, env_image, env_fps):
        self.raw_samples.append((time_sec, raw_pedal))
        self.corrected_samples.append((time_sec, corrected_pedal))

        while self.raw_samples and time_sec - self.raw_samples[0][0] > self.window_sec:
            self.raw_samples.popleft()
        while self.corrected_samples and time_sec - self.corrected_samples[0][0] > self.window_sec:
            self.corrected_samples.popleft()

        raw_x = [sample[0] for sample in self.raw_samples]
        raw_y = [sample[1] for sample in self.raw_samples]
        corrected_x = [sample[0] for sample in self.corrected_samples]
        corrected_y = [sample[1] for sample in self.corrected_samples]

        self.raw_line.set_data(raw_x, raw_y)
        self.corrected_line.set_data(corrected_x, corrected_y)

        x_min = max(time_sec - self.window_sec, 0.0)
        x_max = max(time_sec + 1.0, x_min + self.window_sec)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_facecolor("#e9f7df" if environment_enable >= 0.5 else "#fdeaea")
        self.ax.set_title(
            f"Jetson Real-time Pedal Control    t={time_sec:.2f}s   label={label}   "
            f"lambda={lambda_value:.3f}   E={int(environment_enable)}({env_prediction})   "
            f"img={env_image}   env_fps={env_fps:.1f}"
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.record_frame()
        plt.pause(0.001)

    def record_frame(self):
        if self.record_path is None:
            return

        width, height = self.fig.canvas.get_width_height()
        image = np.asarray(self.fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        if self.writer is None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.record_path), fourcc, self.record_fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.record_path}")

        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(
        description="Jetson real-time UDP pedal receiver, Gaussian controller, and Arduino motor sender."
    )
    parser.add_argument("--udp-ip", default=DEFAULT_UDP_IP, help=f"UDP bind IP. Default: {DEFAULT_UDP_IP}")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help=f"UDP port. Default: {DEFAULT_UDP_PORT}")
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT, help=f"Arduino serial port. Default: {DEFAULT_SERIAL_PORT}")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help=f"Arduino baudrate. Default: {DEFAULT_BAUDRATE}")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH), help=f"Baseline state path. Default: {DEFAULT_STATE_PATH}")
    parser.add_argument(
        "--initial-state",
        default=str(DEFAULT_INITIAL_STATE_PATH),
        help=f"Initial/next-start baseline state path. Default: {DEFAULT_INITIAL_STATE_PATH}",
    )
    parser.add_argument(
        "--warmup-state",
        default=str(DEFAULT_WARMUP_STATE_PATH),
        help=f"Current-start warmup state path. Default: {DEFAULT_WARMUP_STATE_PATH}",
    )
    parser.add_argument(
        "--transition-samples",
        type=float,
        default=TRANSITION_SAMPLE_COUNT,
        help=f"Sample count needed to fully trust the warmup distribution. Default: {TRANSITION_SAMPLE_COUNT:g}",
    )
    parser.add_argument(
        "--keep-warmup-on-start",
        action="store_true",
        help="Continue from an existing warmup state instead of clearing it at program start.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not open Arduino serial; print motor commands only.")
    parser.add_argument("--log-csv", default=DEFAULT_LOG_CSV, help=f"Runtime log CSV path. Default: {DEFAULT_LOG_CSV}")
    parser.add_argument(
        "--transition-csv",
        default=DEFAULT_TRANSITION_CSV,
        help=f"A/B/C baseline transition history CSV path. Default: {DEFAULT_TRANSITION_CSV}",
    )
    parser.add_argument("--no-transition-csv", action="store_true", help="Disable A/B/C baseline transition history logging.")
    parser.add_argument("--plot", action="store_true", help="Show live matplotlib plot for raw vs corrected pedal.")
    parser.add_argument("--plot-window", type=float, default=DEFAULT_PLOT_WINDOW_SEC, help=f"Visible plot window in seconds. Default: {DEFAULT_PLOT_WINDOW_SEC}")
    parser.add_argument("--record-plot", default=DEFAULT_PLOT_VIDEO, help=f"MP4 path for plot recording when --plot is enabled. Default: {DEFAULT_PLOT_VIDEO}")
    parser.add_argument("--no-record-plot", action="store_true", help="Disable plot recording.")
    parser.add_argument("--record-fps", type=float, default=20.0, help="Playback FPS for plot recording. Default: 20.0")
    parser.add_argument("--env-config", default=str(DEFAULT_ENV_CONFIG), help=f"Environment model config path. Default: {DEFAULT_ENV_CONFIG}")
    parser.add_argument("--samples-dir", default=str(DEFAULT_SAMPLES_DIR), help=f"Sample image directory for environment input. Default: {DEFAULT_SAMPLES_DIR}")
    parser.add_argument("--env-device", default=None, help="Optional torch device for environment classifier, e.g. cuda or cpu.")
    parser.add_argument("--env-warmup", type=int, default=DEFAULT_ENV_WARMUP, help=f"Environment model warmup passes. Default: {DEFAULT_ENV_WARMUP}")
    parser.add_argument("--infer-every", type=int, default=DEFAULT_INFER_EVERY, help=f"Run environment inference every N camera frames. Default: {DEFAULT_INFER_EVERY}")
    parser.add_argument(
        "--env-switch-threshold",
        type=int,
        default=ENV_SWITCH_THRESHOLD,
        help=f"Consecutive opposite environment predictions needed to switch E. Default: {ENV_SWITCH_THRESHOLD}",
    )
    parser.add_argument("--no-video", action="store_true", help="Disable USB camera RTP/JPEG video streaming.")
    parser.add_argument("--video-device", default=DEFAULT_CAMERA_DEVICE, help=f"USB camera device. Default: {DEFAULT_CAMERA_DEVICE}")
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH, help=f"Video stream width. Default: {DEFAULT_VIDEO_WIDTH}")
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT, help=f"Video stream height. Default: {DEFAULT_VIDEO_HEIGHT}")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS, help=f"Video stream FPS. Default: {DEFAULT_VIDEO_FPS}")
    parser.add_argument("--video-udp-host", help="Laptop IP address to receive the RTP/JPEG video stream.")
    parser.add_argument("--video-udp-port", type=int, default=DEFAULT_VIDEO_UDP_PORT, help=f"Video UDP port. Default: {DEFAULT_VIDEO_UDP_PORT}")
    parser.add_argument(
        "--telemetry-udp-port",
        type=int,
        default=DEFAULT_TELEMETRY_UDP_PORT,
        help=f"UDP port for corrected pedal telemetry to --video-udp-host. Default: {DEFAULT_TELEMETRY_UDP_PORT}",
    )
    parser.add_argument("--no-telemetry", action="store_true", help="Disable corrected pedal telemetry UDP output.")
    parser.add_argument("--video-display", action="store_true", help="Also show the camera preview on the Jetson display.")
    parser.add_argument("--video-windowed", action="store_true", help="Show Jetson preview in a normal window instead of fullscreen.")
    parser.add_argument("--video-window-width", type=int, default=960, help="Jetson preview window width. Default: 960")
    parser.add_argument("--video-window-height", type=int, default=540, help="Jetson preview window height. Default: 540")
    parser.add_argument("--video-flip", action="store_true", help="Flip the video horizontally.")
    parser.add_argument(
        "--video-state",
        type=int,
        choices=(0, 1),
        default=None,
        help="Fixed ON/OFF overlay state for testing. Overrides controller state when set.",
    )
    parser.add_argument(
        "--video-overlay-source",
        choices=("current-env", "event-env"),
        default="current-env",
        help="State source for video overlay when --video-state is not set. Default: current-env.",
    )
    args = parser.parse_args()
    if args.infer_every <= 0:
        raise ValueError("--infer-every must be 1 or greater.")
    if args.env_switch_threshold <= 0:
        raise ValueError("--env-switch-threshold must be 1 or greater.")

    environment_model, environment_config, environment_device = load_classifier(Path(args.env_config), args.env_device)
    warmup_model(environment_model, environment_config, environment_device, args.env_warmup)
    environment_state = RealtimeEnvironmentState()
    mode_state = ModeState(1)
    emergency_overlay_state = EmergencyOverlayState()
    video_overlay_state = VideoOverlayState(args.video_state if args.video_state is not None else 0)
    video_streamer = None
    controller = PedalController(
        args.state,
        args.initial_state,
        args.warmup_state,
        args.transition_samples,
        reset_warmup_on_start=not args.keep_warmup_on_start,
        transition_history_path=None if args.no_transition_csv else args.transition_csv,
    )
    sock = open_udp_socket(args.udp_ip, args.udp_port)
    telemetry_sock = None if args.no_telemetry or not args.video_udp_host else open_telemetry_socket()
    ser = None if args.dry_run else open_serial(args.serial_port, args.baudrate)
    log_csv_path = init_log_csv(args.log_csv)
    plot_record_path = None if args.no_record_plot else args.record_plot
    plotter = LivePlotter(args.plot_window, plot_record_path, args.record_fps) if args.plot else None
    if not args.no_video:
        if not args.video_udp_host:
            raise ValueError("--video-udp-host is required unless --no-video is set.")
        video_streamer = VideoStreamer(
            args.video_device,
            args.video_width,
            args.video_height,
            args.video_fps,
            args.video_udp_host,
            args.video_udp_port,
            video_overlay_state,
            no_display=not args.video_display,
            windowed=args.video_windowed,
            window_width=args.video_window_width,
            window_height=args.video_window_height,
            flip=args.video_flip,
            environment_state=environment_state,
            model=environment_model,
            config=environment_config,
            model_device=environment_device,
            infer_every=args.infer_every,
            switch_threshold=args.env_switch_threshold,
            update_overlay_from_inference=args.video_state is None and args.video_overlay_source == "current-env",
            mode_state=mode_state,
            emergency_overlay_state=emergency_overlay_state,
        )
        video_streamer.start()

    print(f"Listening UDP on {args.udp_ip}:{args.udp_port}")
    if args.dry_run:
        print("Dry run: Arduino serial output is disabled.")
    else:
        print(f"Arduino serial connected: {args.serial_port} @ {args.baudrate}")
    print(f"Environment model: {Path(args.env_config).resolve()}")
    print(f"Environment device: {environment_device}")
    print(f"Environment input: live camera frames ({args.video_device})")
    print(f"Environment switch threshold: {args.env_switch_threshold}")
    print(f"Environment inference: every {args.infer_every} frame(s)")
    if video_streamer is None:
        print("Video streaming: disabled")
    else:
        print(f"Video overlay source: {'fixed ' + str(args.video_state) if args.video_state is not None else args.video_overlay_source}")
    print(f"Initial baseline A: {controller.initial_state_path}")
    print(f"Warmup baseline B: {controller.warmup_state_path}")
    print(f"Active baseline C: {controller.state_path}")
    print(f"Transition samples: {controller.transition_samples:g}")
    if controller.transition_history_path is not None:
        print(f"Transition history: {controller.transition_history_path}")
    print(f"Runtime log: {log_csv_path}")
    if telemetry_sock is not None:
        print(f"Corrected pedal telemetry: udp://{args.video_udp_host}:{args.telemetry_udp_port} format=time_sec,corrected_pedal")
    else:
        print("Corrected pedal telemetry: disabled")
    if plotter is not None:
        print("Live plot: enabled (press q or Esc in the plot window to stop)")
        if plot_record_path is not None:
            print(f"Plot recording: {plot_record_path}")
    print("Stop: Ctrl+C")

    try:
        while True:
            if plotter is not None and plotter.stop_requested:
                print("Plot requested stop.")
                break
            data, address = sock.recvfrom(1024)
            message = data.decode("utf-8", errors="replace")

            try:
                time_sec, raw_pedal, mode, emergency = parse_udp_message(message)
            except ValueError as exc:
                print(f"ignored packet from {address}: {exc}")
                continue

            mode_state.set(mode)
            env_snapshot = environment_state.current()
            result = controller.process_sample(time_sec, raw_pedal, env_snapshot, emergency)
            emergency_overlay_state.set(result["emergency_active"])
            if args.video_state is None:
                if args.video_overlay_source == "event-env":
                    video_overlay_state.set(1 if result["environment_enable"] >= 0.5 else 0)
                else:
                    video_overlay_state.set(1 if result["current_environment_enable"] >= 0.5 else 0)
            motor_command = "DRY"
            if ser is not None:
                motor_command = send_motor_command(
                    ser,
                    result["corrected_pedal"],
                    result["arduino_environment_enable"],
                )
            else:
                motor_command = motor_command_with_environment(
                    result["corrected_pedal"],
                    result["arduino_environment_enable"],
                )

            append_log_csv(log_csv_path, address, result, motor_command)
            send_corrected_pedal_telemetry(
                telemetry_sock,
                args.video_udp_host,
                args.telemetry_udp_port,
                result["time_sec"],
                result["corrected_pedal"],
            )
            if plotter is not None:
                plotter.update(
                    result["time_sec"],
                    result["raw_pedal"],
                    result["corrected_pedal"],
                    result["label"],
                    result["lambda"],
                    result["environment_enable"],
                    result["environment_prediction"],
                    result["environment_image"],
                    result["environment_fps"],
                )

            z_text = "n/a" if result["z"] is None else f"{result['z']:.2f}"
            slope_text = "n/a" if result["slope"] is None else f"{result['slope']:.2f}"
            print(
                f"from={address[0]}:{address[1]} "
                f"t={time_sec:6.2f}s raw={result['raw_pedal']:6.2f}% "
                f"u={result['corrected_pedal']:6.2f}% cmd={motor_command:>4} "
                f"Ecur={int(result['current_environment_enable'])}({result['current_environment_prediction']}) "
                f"Elatch={int(result['environment_enable'])}({result['environment_prediction']}) "
                f"img={result['environment_image']} env_fps={result['environment_fps']:5.1f} "
                f"block={int(bool(result['blocked_until_event_end']))} "
                f"label={result['label']:<7} z={z_text:>5} slope={slope_text:>6} "
                f"k={result['k']:.3f} lambda={result['lambda']:.3f} "
                f"nB={result['warmup_state']['sample_count']:.2f} "
                f"alpha={result['transition_alpha']:.3f} action={result['action']}"
            )

    except KeyboardInterrupt:
        print("Interrupted by user.")
        if ser is not None:
            send_line(ser, "STOP")
            print("Sent STOP to Arduino.")
    finally:
        controller.finalize_drive()
        sock.close()
        if telemetry_sock is not None:
            telemetry_sock.close()
        if video_streamer is not None:
            video_streamer.stop()
        if plotter is not None:
            plotter.close()
        if ser is not None:
            time.sleep(0.2)
            ser.close()


if __name__ == "__main__":
    main()
