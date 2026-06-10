import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def load_pedal_analysis():
    module_path = Path(__file__).resolve().parent / "02_pedal_analysis.py"
    spec = importlib.util.spec_from_file_location("pedal_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pa = load_pedal_analysis()


DEFAULT_SPEED = 1.0
DISPLAY_FPS = 20
ENVIRONMENT_ENABLE = 1.0


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def control_gain_from_z(z):
    z_abs = abs(float(z))
    if z_abs < pa.NORMAL_UPDATE_Z:
        return 0.0
    if z_abs >= pa.OUTLIER_Z:
        return 1.0

    ratio = (z_abs - pa.NORMAL_UPDATE_Z) / (pa.OUTLIER_Z - pa.NORMAL_UPDATE_Z)
    return ratio ** 3


def control_lambda(reference_slope, current_slope, z, env_enable=ENVIRONMENT_ENABLE):
    if current_slope is None or not np.isfinite(current_slope) or current_slope <= 0:
        return 1.0, 0.0
    if reference_slope is None or not np.isfinite(reference_slope) or reference_slope <= 0:
        return 1.0, 0.0

    ratio = clamp(reference_slope / current_slope, 0.0, 1.0)
    k = control_gain_from_z(z)
    lam = 1.0 - env_enable * k * (1.0 - ratio)
    return clamp(lam, 0.0, 1.0), k


def load_series(csv_path):
    df = pa.load_data(csv_path)
    time = df["time_sec"].to_numpy()
    pedal = df["pedal_percent"].to_numpy()
    events = pa.detect_events(time, pedal)
    event_table = pa.build_event_table(time, pedal, events)
    return time, pedal, event_table


def build_event_lookup(event_table, sample_count):
    lookup = [None] * sample_count
    for row in event_table.itertuples(index=False):
        for idx in range(int(row.start_idx), int(row.end_idx) + 1):
            lookup[idx] = row
    return lookup


def corrected_sample(time, pedal, state, event_row, sample_idx, can_judge):
    raw_value = float(pedal[sample_idx])
    if event_row is None or not can_judge or not pa.state_ready(state):
        return raw_value, "OK", 0.0, 1.0, 0.0, None

    current_slope = pa.representative_slope(
        time,
        pedal,
        int(event_row.start_idx),
        sample_idx,
    )
    if current_slope is None or not np.isfinite(current_slope):
        return raw_value, "OK", 0.0, 1.0, 0.0, None

    mean = state["mean"]
    sigma = state["sigma"]
    z = pa.z_score(current_slope, mean, sigma)
    label = pa.classify_z(current_slope, mean, sigma)

    lam, k = control_lambda(mean, current_slope, z)
    return raw_value * lam, label, z, lam, k, current_slope


def update_state_at_event_end(state, event_row, using_saved_state, split_time):
    if event_row.slope is None or not np.isfinite(event_row.slope):
        return "ignored", None, None, 0.0

    if not using_saved_state and event_row.end_time <= split_time:
        pa.update_baseline_state(state, event_row.slope, 1.0)
        return "warm-up learned", "WARMUP", None, 1.0

    should_judge = using_saved_state or event_row.start_time >= split_time
    if not should_judge or not pa.state_ready(state):
        return "not judged", None, None, 0.0

    mean = state["mean"]
    sigma = state["sigma"]
    z = pa.z_score(event_row.slope, mean, sigma)
    label = pa.classify_z(event_row.slope, mean, sigma)
    weight, action = pa.baseline_update_weight(event_row.slope, mean, sigma)
    if weight > 0:
        pa.update_baseline_state(state, event_row.slope, weight)
    return action, label, z, weight


def create_animation(csv_path, state_path, fresh=False, speed=DEFAULT_SPEED):
    time, pedal, event_table = load_series(csv_path)
    event_lookup = build_event_lookup(event_table, len(time))

    saved_state = None if fresh else pa.load_baseline_state(state_path)
    using_saved_state = saved_state is not None
    state = saved_state if using_saved_state else pa.empty_baseline_state()

    corrected = np.full_like(pedal, np.nan, dtype=float)
    processed_until = {"idx": -1}
    completed_events = set()
    status = {
        "label": "READY",
        "z": None,
        "lambda": 1.0,
        "k": 0.0,
        "slope": None,
        "message": "waiting for samples",
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    corrected_line, = ax.plot(
        [],
        [],
        color="#1f77b4",
        linewidth=2.4,
        alpha=0.88,
        label="corrected pedal",
        zorder=2,
    )
    raw_line, = ax.plot(
        [],
        [],
        color="#d62728",
        linewidth=2.0,
        label="raw pedal",
        zorder=3,
    )
    info_text = ax.text(
        0.015,
        0.965,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.86},
    )

    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(-2, max(10.0, float(np.max(pedal))) * 1.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pedal (%)")
    ax.set_title("Streaming Raw vs Corrected Pedal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    if not using_saved_state:
        ax.axvline(pa.SPLIT_TIME_SEC, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8)

    if using_saved_state:
        print(
            f"Loaded baseline state: mean={state['mean']:.3f}, "
            f"sigma={state['sigma']:.3f}, n={state['sample_count']:.2f}"
        )
        print("Judging and correction are active from the beginning.")
    else:
        print(f"Fresh mode: 0-{pa.SPLIT_TIME_SEC:g}s warm-up, then correction can start.")

    def process_to(target_idx):
        for idx in range(processed_until["idx"] + 1, target_idx + 1):
            event_row = event_lookup[idx]
            can_judge = using_saved_state or float(time[idx]) >= pa.SPLIT_TIME_SEC
            corrected_value, label, z, lam, k, slope = corrected_sample(
                time,
                pedal,
                state,
                event_row,
                idx,
                can_judge,
            )
            corrected[idx] = corrected_value

            status["label"] = label
            status["z"] = z
            status["lambda"] = lam
            status["k"] = k
            status["slope"] = slope
            status["message"] = "streaming"

            if event_row is not None and int(event_row.end_idx) == idx and event_row.event_no not in completed_events:
                completed_events.add(event_row.event_no)
                action, final_label, final_z, weight = update_state_at_event_end(
                    state,
                    event_row,
                    using_saved_state,
                    pa.SPLIT_TIME_SEC,
                )
                pa.save_baseline_state(state_path, state)
                if final_label is not None:
                    status["label"] = final_label
                    status["z"] = final_z
                status["message"] = f"event {event_row.event_no:02d}: {action}, weight={weight:g}, saved"

        processed_until["idx"] = target_idx

    duration_sec = float(time[-1] - time[0])
    playback_sec = duration_sec / max(speed, 0.1)
    total_frames = max(2, int(np.ceil(playback_sec * DISPLAY_FPS)) + 1)
    interval_ms = int(1000 / DISPLAY_FPS)

    def frame_update(frame_no):
        elapsed_display_sec = frame_no / DISPLAY_FPS
        current_time = float(time[0] + elapsed_display_sec * speed)
        if frame_no >= total_frames - 1:
            current_time = float(time[-1])

        frame_idx = int(np.searchsorted(time, current_time, side="right") - 1)
        frame_idx = clamp(frame_idx, 0, len(time) - 1)
        process_to(frame_idx)

        visible = slice(0, frame_idx + 1)
        raw_line.set_data(time[visible], pedal[visible])
        corrected_line.set_data(time[visible], corrected[visible])

        z_text = "n/a" if status["z"] is None else f"{status['z']:.2f}"
        slope_text = "n/a" if status["slope"] is None else f"{status['slope']:.2f}"
        info_text.set_text(
            "\n".join(
                [
                    f"t = {time[frame_idx]:.1f}s  (speed x{speed:g})",
                    f"state: {status['label']} | z = {z_text} | k = {status['k']:.3f} | lambda = {status['lambda']:.3f}",
                    f"current slope = {slope_text}",
                    f"mean = {state['mean']:.3f}, sigma = {state['sigma']:.3f}, n = {state['sample_count']:.2f}",
                    status["message"],
                ]
            )
        )
        return raw_line, corrected_line, info_text

    anim = FuncAnimation(
        fig,
        frame_update,
        frames=total_frames,
        interval=interval_ms,
        repeat=False,
        blit=False,
    )
    fig.tight_layout()
    return fig, anim


def main():
    parser = argparse.ArgumentParser(
        description="Animate streaming raw pedal input and OUTLIER-corrected pedal input."
    )
    parser.add_argument("--csv", default=pa.CSV_NAME, help=f"CSV path. Default: {pa.CSV_NAME}")
    parser.add_argument("--state", default=pa.STATE_NAME, help=f"Baseline state path. Default: {pa.STATE_NAME}")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved state and simulate first-use warm-up.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help=f"Animation speed multiplier. Default: {DEFAULT_SPEED:g}")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = base_dir / state_path

    fig, anim = create_animation(csv_path, state_path, fresh=args.fresh, speed=args.speed)
    plt.show()


if __name__ == "__main__":
    main()
