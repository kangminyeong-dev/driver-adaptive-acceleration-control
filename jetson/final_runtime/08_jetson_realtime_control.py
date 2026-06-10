import argparse
import json
import socket
import threading
import time
from collections import deque
import csv
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_runtime import load_classifier, predict_image

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

PEDAL_THRESHOLD = 5.0
HIGH_PEDAL_THRESHOLD = 80.0
HIGH_PEDAL_HOLD_SEC = 2.0
SLOPE_STEP_COUNT = 4
ENV_SWITCH_THRESHOLD = 5
NORMAL_UPDATE_Z = 2.0
OUTLIER_Z = 3.0
WARN_UPDATE_WEIGHT = 0.5
MIN_SIGMA = 0.1
TRANSITION_SAMPLE_COUNT = 60.0
ENVIRONMENT_ENABLE_DEFAULT = 1.0
FORCE_ENVIRONMENT_ENABLE_ON = True


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
    if len(parts) != 2:
        raise ValueError(f"Expected 'time_sec,pedal_percent', got: {message!r}")
    return float(parts[0]), float(parts[1])


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


def send_motor_command(ser, corrected_pedal):
    command = motor_command_from_pedal(corrected_pedal)
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

    def process_sample(self, time_sec, raw_pedal, env_snapshot):
        raw_pedal = clamp(raw_pedal, 0.0, 100.0)
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

            if self.block_until_event_end:
                corrected_pedal = 0.0
                action = "blocked_until_event_end"
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
    args = parser.parse_args()

    environment_source = SampleEnvironmentClassifier(args.env_config, args.samples_dir, args.env_device)
    environment_monitor = BackgroundEnvironmentMonitor(environment_source)
    controller = PedalController(
        args.state,
        args.initial_state,
        args.warmup_state,
        args.transition_samples,
        reset_warmup_on_start=not args.keep_warmup_on_start,
        transition_history_path=None if args.no_transition_csv else args.transition_csv,
    )
    sock = open_udp_socket(args.udp_ip, args.udp_port)
    ser = None if args.dry_run else open_serial(args.serial_port, args.baudrate)
    log_csv_path = init_log_csv(args.log_csv)
    plot_record_path = None if args.no_record_plot else args.record_plot
    plotter = LivePlotter(args.plot_window, plot_record_path, args.record_fps) if args.plot else None
    environment_monitor.start()

    print(f"Listening UDP on {args.udp_ip}:{args.udp_port}")
    if args.dry_run:
        print("Dry run: Arduino serial output is disabled.")
    else:
        print(f"Arduino serial connected: {args.serial_port} @ {args.baudrate}")
    print(f"Environment model: {environment_source.config_path}")
    print(f"Environment samples: {environment_source.samples_dir} ({len(environment_source.image_paths)} images)")
    print(f"Environment device: {environment_source.device}")
    print("Environment loop: running independently")
    print(f"Initial baseline A: {controller.initial_state_path}")
    print(f"Warmup baseline B: {controller.warmup_state_path}")
    print(f"Active baseline C: {controller.state_path}")
    print(f"Transition samples: {controller.transition_samples:g}")
    if controller.transition_history_path is not None:
        print(f"Transition history: {controller.transition_history_path}")
    print(f"Runtime log: {log_csv_path}")
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
                time_sec, raw_pedal = parse_udp_message(message)
            except ValueError as exc:
                print(f"ignored packet from {address}: {exc}")
                continue

            env_snapshot = environment_monitor.current()
            result = controller.process_sample(time_sec, raw_pedal, env_snapshot)
            motor_command = "DRY"
            if ser is not None:
                motor_command = send_motor_command(ser, result["corrected_pedal"])
            else:
                motor_command = motor_command_from_pedal(result["corrected_pedal"])

            append_log_csv(log_csv_path, address, result, motor_command)
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
        environment_monitor.stop()
        if plotter is not None:
            plotter.close()
        if ser is not None:
            time.sleep(0.2)
            ser.close()


if __name__ == "__main__":
    main()
