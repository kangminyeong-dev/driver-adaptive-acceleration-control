import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch


CSV_NAME = "pedal_input_sequence_deadzone_pull_rounded_1dp_120s.csv"
STATE_NAME = "pedal_baseline_state.json"
SPLIT_TIME_SEC = 60.0
PEDAL_THRESHOLD = 5.0
SLOPE_STEP_COUNT = 4
ANIMATION_INTERVAL_MS = round(80 / 0.75)
NORMAL_UPDATE_Z = 2.0
OUTLIER_Z = 3.0
WARN_UPDATE_WEIGHT = 0.25
MIN_SIGMA = 0.1


def load_data(csv_path):
    """Load pedal CSV and sort by time."""
    df = pd.read_csv(csv_path)
    required_columns = {"time_sec", "pedal_percent"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df = df[["time_sec", "pedal_percent"]].copy()
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="raise")
    df["pedal_percent"] = pd.to_numeric(df["pedal_percent"], errors="raise")
    return df.sort_values("time_sec").reset_index(drop=True)


def detect_events(time, pedal, threshold=PEDAL_THRESHOLD):
    """Return contiguous pedal-on events where pedal >= threshold."""
    events = []
    start_idx = None

    for idx, value in enumerate(pedal):
        if value >= threshold and start_idx is None:
            start_idx = idx
        elif value < threshold and start_idx is not None:
            events.append((start_idx, idx - 1))
            start_idx = None

    if start_idx is not None:
        events.append((start_idx, len(pedal) - 1))

    return events


def smooth_pedal(pedal, window=3):
    """Small centered moving average for the left-hand time plot."""
    if window <= 1:
        return pedal.copy()
    return pd.Series(pedal).rolling(window, center=True, min_periods=1).mean().to_numpy()


def representative_slope(time, pedal, start_idx, end_idx, step_count=SLOPE_STEP_COUNT):
    """
    Maximum rolling average change rate inside one event.

    For each moving window t-4, t-3, t-2, t-1, t, calculate the
    average of the 4 sequential change rates. The event's representative
    slope is the largest such rolling value.
    """
    if end_idx - start_idx < step_count:
        return None

    best_slope = None
    for idx in range(start_idx + step_count, end_idx + 1):
        window_start = idx - step_count
        dt = np.diff(time[window_start : idx + 1])
        dp = np.diff(pedal[window_start : idx + 1])
        valid = dt > 0

        if not np.any(valid):
            continue

        rolling_slope = float(np.mean(dp[valid] / dt[valid]))
        if best_slope is None or rolling_slope > best_slope:
            best_slope = rolling_slope

    return best_slope


def build_event_table(time, pedal, events, split_time=SPLIT_TIME_SEC):
    """Build one row per detected pedal event."""
    rows = []
    for event_no, (start_idx, end_idx) in enumerate(events, start=1):
        slope = representative_slope(time, pedal, start_idx, end_idx)
        start_time = float(time[start_idx])
        end_time = float(time[end_idx])

        if end_time <= split_time:
            period = "baseline"
        elif start_time >= split_time:
            period = "after"
        else:
            period = "crossing"

        rows.append(
            {
                "event_no": event_no,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_time,
                "end_time": end_time,
                "max_pedal": float(np.max(pedal[start_idx : end_idx + 1])),
                "slope": slope,
                "period": period,
            }
        )

    return pd.DataFrame(rows)


def fit_gaussian(samples):
    """Fit mu/sigma for display-only sample arrays."""
    samples = np.asarray([sample for sample in samples if sample is not None], dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        raise ValueError("No valid baseline slope samples.")

    mu = float(np.mean(samples))
    if len(samples) == 1:
        sigma = max(abs(mu) * 0.05, MIN_SIGMA)
    else:
        sigma = max(float(np.std(samples, ddof=1)), MIN_SIGMA)

    return mu, sigma, samples


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

    if n <= 0:
        variance = 0.0
        sigma = MIN_SIGMA
    elif n <= 1:
        sigma = max(abs(mean) * 0.05, MIN_SIGMA)
        variance = sigma * sigma
    else:
        variance = max(m2 / (n - 1.0), MIN_SIGMA * MIN_SIGMA)
        sigma = float(np.sqrt(variance))

    state["sample_count"] = n
    state["mean"] = mean
    state["m2"] = m2
    state["variance"] = float(variance)
    state["sigma"] = float(sigma)
    state["learned_event_count"] = int(state.get("learned_event_count", 0))
    return state


def update_baseline_state(state, value, weight=1.0):
    """Weighted online mean/variance update using Welford's method."""
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

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    if "samples" in state:
        migrated = empty_baseline_state()
        for sample in np.asarray(state["samples"], dtype=float):
            if np.isfinite(sample):
                update_baseline_state(migrated, sample, 1.0)
        return migrated

    required = {"sample_count", "mean", "m2"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"{state_path} is missing saved fields: {sorted(missing)}")

    return refresh_state_stats(state)


def save_baseline_state(state_path, state):
    state = refresh_state_stats(dict(state))
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state


def normal_pdf(x, mu, sigma):
    """Normal distribution PDF without requiring scipy."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def z_score(value, mu, sigma):
    return (value - mu) / sigma


def classify_z(value, mu, sigma):
    z = abs(z_score(value, mu, sigma))
    if z >= OUTLIER_Z:
        return "OUTLIER"
    if z >= NORMAL_UPDATE_Z:
        return "WARN"
    return "OK"


def sigma_band_color(value, mu, sigma):
    """Color for after-60s events based on absolute sigma distance."""
    z = abs(z_score(value, mu, sigma))
    if z < 1:
        return "#1f9d55", "0-1 sigma"
    if z < 2:
        return "#7ed957", "1-2 sigma"
    if z < 3:
        return "#c6f542", "2-3 sigma"
    return "#d62728", "outside 3 sigma"


def baseline_update_weight(value, mu, sigma):
    """
    Return the online-update weight for one judged event.

    OK events count as one normal sample. WARN events count as a fractional
    sample, so they can influence the baseline without dominating it.
    """
    z = abs(z_score(value, mu, sigma))
    if z < NORMAL_UPDATE_Z:
        return 1.0, "learned"
    if z < OUTLIER_Z:
        return WARN_UPDATE_WEIGHT, f"weak learned ({WARN_UPDATE_WEIGHT:g}x)"
    return 0.0, "held out"


def visible_after_slopes(event_table, current_idx, time, pedal):
    """Completed after-60s events plus the currently active after-60s event."""
    rows = []

    for row in event_table.itertuples(index=False):
        if row.period != "after":
            continue

        slope = None
        state = None

        if row.end_idx <= current_idx:
            slope = row.slope
            state = "complete"
        elif row.start_idx <= current_idx < row.end_idx:
            slope = representative_slope(time, pedal, row.start_idx, current_idx)
            state = "current"

        if slope is not None and np.isfinite(slope):
            rows.append(
                {
                    "event_no": row.event_no,
                    "start_time": row.start_time,
                    "slope": float(slope),
                    "state": state,
                }
            )

    return rows


def visible_baseline_slopes(event_table, current_idx):
    """Baseline events completed by the current animation frame."""
    samples = []
    for row in event_table.itertuples(index=False):
        if row.period == "baseline" and row.end_idx <= current_idx and row.slope is not None:
            samples.append(float(row.slope))
    return np.asarray(samples, dtype=float)


def distribution_axis_range(mu, sigma, baseline_samples, after_points=None):
    values = [mu - 4 * sigma, mu + 4 * sigma, *baseline_samples]
    if after_points:
        values.extend(point["slope"] for point in after_points)

    x_min = float(np.min(values))
    x_max = float(np.max(values))
    padding = max((x_max - x_min) * 0.12, sigma * 0.6, 0.5)
    return x_min - padding, x_max + padding


def draw_waiting_distribution(ax, sample_count):
    ax.text(
        0.5,
        0.52,
        f"Waiting for baseline events\ncompleted samples: {sample_count}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=13,
        color="#555555",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Representative slope (pedal %/s)")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.18)


def draw_distribution(ax, mu, sigma, baseline_samples, after_points=None, locked=False):
    after_points = after_points or []
    x_min, x_max = distribution_axis_range(mu, sigma, baseline_samples, after_points)
    x = np.linspace(x_min, x_max, 500)
    y = normal_pdf(x, mu, sigma)
    y_peak = float(np.max(y))

    bands = [
        (x_min, mu - 3 * sigma, "#d62728", 0.10),
        (mu - 3 * sigma, mu - 2 * sigma, "#c6f542", 0.13),
        (mu - 2 * sigma, mu - sigma, "#7ed957", 0.13),
        (mu - sigma, mu + sigma, "#1f9d55", 0.12),
        (mu + sigma, mu + 2 * sigma, "#7ed957", 0.13),
        (mu + 2 * sigma, mu + 3 * sigma, "#c6f542", 0.13),
        (mu + 3 * sigma, x_max, "#d62728", 0.10),
    ]
    for lower, upper, color, alpha in bands:
        if upper > lower:
            ax.axvspan(lower, upper, color=color, alpha=alpha, linewidth=0, zorder=0)

    for sigma_no in [1, 2, 3]:
        ax.axvline(mu - sigma_no * sigma, color="gray", linestyle="--", linewidth=0.8, alpha=0.22, zorder=1)
        ax.axvline(mu + sigma_no * sigma, color="gray", linestyle="--", linewidth=0.8, alpha=0.22, zorder=1)

    label = "locked Gaussian" if locked else "forming Gaussian"
    ax.plot(x, y, color="#1f77b4", linewidth=2.5, label=label, zorder=3)
    ax.axvline(mu, color="#1f77b4", linewidth=1.8, label=f"mean = {mu:.2f}", zorder=4)

    sample_y = np.full(len(baseline_samples), y_peak * 0.035)
    ax.scatter(
        baseline_samples,
        sample_y,
        marker="|",
        s=180,
        linewidths=1.4,
        color="#1f77b4",
        alpha=0.70,
        label="0-60s samples",
        zorder=5,
    )

    complete_points = [point for point in after_points if point["state"] == "complete"]
    current_points = [point for point in after_points if point["state"] == "current"]

    if complete_points:
        xs = np.array([point["slope"] for point in complete_points], dtype=float)
        ys = np.maximum(normal_pdf(xs, mu, sigma), y_peak * 0.035)
        ax.scatter(
            xs,
            ys,
            s=75,
            color="#d62728",
            edgecolors="#7f0000",
            linewidths=0.8,
            zorder=5,
            label="after 60s events",
        )

    if current_points:
        xs = np.array([point["slope"] for point in current_points], dtype=float)
        ys = np.maximum(normal_pdf(xs, mu, sigma), y_peak * 0.07)
        ax.scatter(
            xs,
            ys,
            s=180,
            marker="*",
            color="#d62728",
            edgecolors="#7f0000",
            linewidths=1.1,
            zorder=6,
            label="current after-60s event",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_peak * 1.2)
    ax.set_xlabel("Representative slope (pedal %/s)")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.12)
    ax.legend(loc="upper right", fontsize=9)


def create_animation(csv_path, threshold=PEDAL_THRESHOLD, split_time=SPLIT_TIME_SEC, smoothing_window=3):
    df = load_data(csv_path)
    time = df["time_sec"].to_numpy()
    pedal = df["pedal_percent"].to_numpy()
    pedal_smooth = smooth_pedal(pedal, smoothing_window)

    events = detect_events(time, pedal, threshold)
    event_table = build_event_table(time, pedal, events, split_time)
    state_path = Path(csv_path).resolve().parent / STATE_NAME
    saved_state = load_baseline_state(state_path)
    using_saved_state = saved_state is not None
    baseline_state = saved_state if using_saved_state else empty_baseline_state()
    processed_events = set()
    decisions = []

    print(f"Loaded {len(df)} samples from {time.min():.1f}s to {time.max():.1f}s")
    print(f"Detected {len(events)} pedal events")
    if using_saved_state:
        print(
            f"Loaded baseline state: {state_path.name} "
            f"(mean={baseline_state['mean']:.3f}, variance={baseline_state['variance']:.3f}, "
            f"n={baseline_state['sample_count']:.2f})"
        )
        print("Saved baseline exists, so the full CSV is judged from the beginning.")
    else:
        print(f"No saved baseline. Using 0-{split_time:g}s as warm-up, then judging after that.")
    print("State is saved after each completed event update.")

    fig, (ax_time, ax_dist) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Pedal Gaussian Check", fontsize=16, fontweight="bold")

    frame_indices = np.arange(0, len(df), max(1, len(df) // 180))
    if frame_indices[-1] != len(df) - 1:
        frame_indices = np.append(frame_indices, len(df) - 1)

    y_max = max(10.0, float(np.max(pedal)) * 1.12)

    def process_completed_events(current_idx):
        for row in event_table.itertuples(index=False):
            if row.event_no in processed_events or row.end_idx > current_idx:
                continue

            processed_events.add(row.event_no)
            if row.slope is None or not np.isfinite(row.slope):
                continue

            if not using_saved_state and row.end_time <= split_time:
                update_baseline_state(baseline_state, row.slope, 1.0)
                save_baseline_state(state_path, baseline_state)
                print(
                    f"warm-up event {row.event_no:02d}: slope={row.slope:6.2f}, "
                    f"n={baseline_state['sample_count']:.2f}, mean={baseline_state['mean']:.3f}, "
                    f"sigma={baseline_state['sigma']:.3f}"
                )
                continue

            should_judge = using_saved_state or row.start_time >= split_time
            if not should_judge or not state_ready(baseline_state):
                continue

            mu = baseline_state["mean"]
            sigma = baseline_state["sigma"]
            z = z_score(row.slope, mu, sigma)
            label = classify_z(row.slope, mu, sigma)
            weight, action = baseline_update_weight(row.slope, mu, sigma)

            if weight > 0:
                update_baseline_state(baseline_state, row.slope, weight)
            save_baseline_state(state_path, baseline_state)

            decisions.append(
                {
                    "event_no": row.event_no,
                    "start_time": row.start_time,
                    "end_time": row.end_time,
                    "slope": float(row.slope),
                    "state": "complete",
                    "z": z,
                    "label": label,
                    "action": action,
                }
            )
            print(
                f"judge event {row.event_no:02d}: {row.start_time:5.1f}-{row.end_time:5.1f}s, "
                f"slope={row.slope:6.2f}, z={z:6.2f}, {label}, {action}, "
                f"n={baseline_state['sample_count']:.2f}"
            )

    def current_judged_points(current_idx):
        points = list(decisions)
        if not state_ready(baseline_state):
            return points

        for row in event_table.itertuples(index=False):
            if row.start_idx <= current_idx < row.end_idx:
                should_judge = using_saved_state or row.start_time >= split_time
                if not should_judge:
                    break

                slope = representative_slope(time, pedal, row.start_idx, current_idx)
                if slope is not None and np.isfinite(slope):
                    points.append(
                        {
                            "event_no": row.event_no,
                            "start_time": row.start_time,
                            "slope": float(slope),
                            "state": "current",
                        }
                    )
                break
        return points

    def update(frame_no):
        current_idx = int(frame_indices[frame_no])
        current_time = float(time[current_idx])
        process_completed_events(current_idx)

        ax_time.clear()
        ax_dist.clear()

        if state_ready(baseline_state):
            mu = baseline_state["mean"]
            sigma = baseline_state["sigma"]
        else:
            mu = None
            sigma = None

        ax_time.plot(time[: current_idx + 1], pedal[: current_idx + 1], color="#4c78a8", linewidth=1.4, label="raw")
        ax_time.plot(
            time[: current_idx + 1],
            pedal_smooth[: current_idx + 1],
            color="#f58518",
            linewidth=2.0,
            label="smoothed",
        )
        ax_time.axhline(threshold, color="gray", linestyle=":", linewidth=1.4, label=f"threshold {threshold:g}%")
        ax_time.axvline(split_time, color="#2ca02c", linestyle="--", linewidth=2.0, label="60s split")

        for row in event_table.itertuples(index=False):
            if row.start_idx > current_idx:
                continue

            span_end_idx = min(row.end_idx, current_idx)
            t0 = row.start_time
            t1 = float(time[span_end_idx])
            if row.period == "baseline" and not using_saved_state:
                color = "#2ca02c"
            elif row.period == "crossing" and not using_saved_state:
                color = "#9467bd"
            else:
                if row.end_idx <= current_idx:
                    slope = row.slope
                else:
                    slope = representative_slope(time, pedal, row.start_idx, current_idx)

                if slope is None or not np.isfinite(slope):
                    color = "#b0b0b0"
                elif mu is None or sigma is None:
                    color = "#b0b0b0"
                else:
                    color, _ = sigma_band_color(slope, mu, sigma)
            ax_time.axvspan(t0, t1, color=color, alpha=0.10)

        if mu is None or sigma is None:
            draw_waiting_distribution(ax_dist, 0)
        else:
            after_points = current_judged_points(current_idx)
            draw_distribution(
                ax_dist,
                mu,
                sigma,
                [],
                after_points,
                locked=current_time >= split_time or using_saved_state,
            )

        after_points = current_judged_points(current_idx)
        current_after = [point for point in after_points if point["state"] == "current"]
        completed_after = [point for point in after_points if point["state"] == "complete"]
        subtitle = f"t = {current_time:.1f}s | judged events: {len(completed_after)}"
        if current_after and mu is not None and sigma is not None:
            slope = current_after[0]["slope"]
            subtitle += f" + current slope {slope:.2f} ({classify_z(slope, mu, sigma)})"

        ax_time.set_xlim(0, max(time[-1], split_time) * 1.03)
        ax_time.set_ylim(-2, y_max)
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Pedal (%)")
        ax_time.set_title(subtitle, fontsize=12, fontweight="bold")
        ax_time.grid(True, alpha=0.28)
        line_handles, line_labels = ax_time.get_legend_handles_labels()
        sigma_handles = [
            Patch(facecolor="#1f9d55", alpha=0.22, label="after: 0-1 sigma"),
            Patch(facecolor="#7ed957", alpha=0.22, label="after: 1-2 sigma"),
            Patch(facecolor="#c6f542", alpha=0.22, label="after: 2-3 sigma"),
            Patch(facecolor="#d62728", alpha=0.22, label="after: >3 sigma"),
        ]
        ax_time.legend(line_handles + sigma_handles, line_labels + [h.get_label() for h in sigma_handles], loc="upper left", fontsize=8)

        state = "adaptive" if current_time >= split_time or using_saved_state else "warm-up"
        ax_dist.set_title(f"Baseline Gaussian {state} (n={baseline_state['sample_count']:.2f})", fontsize=12, fontweight="bold")

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=ANIMATION_INTERVAL_MS,
        repeat=False,
        blit=False,
    )
    fig.tight_layout()
    return fig, anim


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / CSV_NAME

    fig, anim = create_animation(csv_path)
    plt.show()
