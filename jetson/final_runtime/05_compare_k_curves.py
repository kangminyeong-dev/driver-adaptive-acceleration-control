import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pedal_analysis():
    module_path = Path(__file__).resolve().parent / "02_pedal_analysis.py"
    spec = importlib.util.spec_from_file_location("pedal_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pa = load_pedal_analysis()


OUTPUT_PNG = "outputs/compare_k_curves.png"
OUTPUT_CSV = "outputs/compare_k_curves.csv"
ENVIRONMENT_ENABLE = 1.0


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def control_gain_from_z(z, power):
    z_abs = abs(float(z))
    if z_abs < pa.NORMAL_UPDATE_Z:
        return 0.0
    if z_abs >= pa.OUTLIER_Z:
        return 1.0

    ratio = (z_abs - pa.NORMAL_UPDATE_Z) / (pa.OUTLIER_Z - pa.NORMAL_UPDATE_Z)
    return ratio ** power


def control_lambda(reference_slope, current_slope, z, power):
    if current_slope is None or not np.isfinite(current_slope) or current_slope <= 0:
        return 1.0, 0.0
    if reference_slope is None or not np.isfinite(reference_slope) or reference_slope <= 0:
        return 1.0, 0.0

    ratio = clamp(reference_slope / current_slope, 0.0, 1.0)
    k = control_gain_from_z(z, power)
    lam = 1.0 - ENVIRONMENT_ENABLE * k * (1.0 - ratio)
    return clamp(lam, 0.0, 1.0), k


def load_event_table(csv_path):
    df = pa.load_data(csv_path)
    time = df["time_sec"].to_numpy()
    pedal = df["pedal_percent"].to_numpy()
    events = pa.detect_events(time, pedal)
    event_table = pa.build_event_table(time, pedal, events)
    return df, time, pedal, event_table


def build_event_lookup(event_table, sample_count):
    lookup = [None] * sample_count
    for row in event_table.itertuples(index=False):
        for idx in range(int(row.start_idx), int(row.end_idx) + 1):
            lookup[idx] = row
    return lookup


def update_state_at_event_end(state, event_row, using_saved_state):
    if event_row.slope is None or not np.isfinite(event_row.slope):
        return

    if not using_saved_state and event_row.end_time <= pa.SPLIT_TIME_SEC:
        pa.update_baseline_state(state, event_row.slope, 1.0)
        return

    should_judge = using_saved_state or event_row.start_time >= pa.SPLIT_TIME_SEC
    if not should_judge or not pa.state_ready(state):
        return

    weight, _ = pa.baseline_update_weight(event_row.slope, state["mean"], state["sigma"])
    if weight > 0:
        pa.update_baseline_state(state, event_row.slope, weight)


def simulate_corrected(time, pedal, event_table, event_lookup, initial_state, using_saved_state, power):
    state = pa.refresh_state_stats(dict(initial_state))
    corrected = pedal.astype(float).copy()
    rows = []
    completed_events = set()

    for idx, raw_value in enumerate(pedal):
        event_row = event_lookup[idx]
        can_judge = using_saved_state or float(time[idx]) >= pa.SPLIT_TIME_SEC
        label = "OK"
        z = 0.0
        k = 0.0
        lam = 1.0
        current_slope = np.nan

        if event_row is not None and can_judge and pa.state_ready(state):
            current_slope = pa.representative_slope(time, pedal, int(event_row.start_idx), idx)
            if current_slope is not None and np.isfinite(current_slope):
                z = pa.z_score(current_slope, state["mean"], state["sigma"])
                label = pa.classify_z(current_slope, state["mean"], state["sigma"])
                lam, k = control_lambda(state["mean"], current_slope, z, power)
                corrected[idx] = float(raw_value) * lam

        rows.append(
            {
                "time_sec": float(time[idx]),
                "raw_pedal": float(raw_value),
                f"corrected_power_{power}": float(corrected[idx]),
                f"label_power_{power}": label,
                f"z_power_{power}": z,
                f"k_power_{power}": k,
                f"lambda_power_{power}": lam,
                f"slope_power_{power}": current_slope,
            }
        )

        if event_row is not None and int(event_row.end_idx) == idx and event_row.event_no not in completed_events:
            completed_events.add(event_row.event_no)
            update_state_at_event_end(state, event_row, using_saved_state)

    return corrected, pd.DataFrame(rows)


def save_plot(time, raw, corrected2, corrected3, output_path):
    z_values = np.linspace(0.0, 4.0, 401)
    k2 = [control_gain_from_z(z, 2) for z in z_values]
    k3 = [control_gain_from_z(z, 3) for z in z_values]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

    axes[0].plot(time, corrected2, color="#2ca02c", linewidth=2.0, alpha=0.80, label="corrected k=r^2")
    axes[0].plot(time, corrected3, color="#1f77b4", linewidth=2.0, alpha=0.85, label="corrected k=r^3")
    axes[0].plot(time, raw, color="#d62728", linewidth=1.8, label="raw pedal", zorder=4)
    axes[0].axvline(pa.SPLIT_TIME_SEC, color="#555555", linestyle="--", linewidth=1.1, alpha=0.7)
    axes[0].set_title("Raw vs Corrected Pedal: Quadratic vs Cubic k")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Pedal (%)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(z_values, k2, color="#2ca02c", linewidth=2.2, label="k = r^2")
    axes[1].plot(z_values, k3, color="#1f77b4", linewidth=2.2, label="k = r^3")
    axes[1].axvline(pa.NORMAL_UPDATE_Z, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].axvline(pa.OUTLIER_Z, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_title("Control Gain Curve")
    axes[1].set_xlabel("|z|")
    axes[1].set_ylabel("k")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare quadratic and cubic control gain curves.")
    parser.add_argument("--csv", default=pa.CSV_NAME, help=f"CSV path. Default: {pa.CSV_NAME}")
    parser.add_argument("--state", default=pa.STATE_NAME, help=f"Baseline state path. Default: {pa.STATE_NAME}")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved state and simulate first-use warm-up.")
    parser.add_argument("--output", default=OUTPUT_PNG, help=f"Output plot path. Default: {OUTPUT_PNG}")
    parser.add_argument("--csv-output", default=OUTPUT_CSV, help=f"Output CSV path. Default: {OUTPUT_CSV}")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = base_dir / state_path

    df, time, pedal, event_table = load_event_table(csv_path)
    event_lookup = build_event_lookup(event_table, len(time))
    saved_state = None if args.fresh else pa.load_baseline_state(state_path)
    using_saved_state = saved_state is not None
    initial_state = saved_state if using_saved_state else pa.empty_baseline_state()

    corrected2, rows2 = simulate_corrected(time, pedal, event_table, event_lookup, initial_state, using_saved_state, 2)
    corrected3, rows3 = simulate_corrected(time, pedal, event_table, event_lookup, initial_state, using_saved_state, 3)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    csv_output_path = Path(args.csv_output)
    if not csv_output_path.is_absolute():
        csv_output_path = base_dir / csv_output_path

    merged = df.copy()
    merged["corrected_k2"] = corrected2
    merged["corrected_k3"] = corrected3
    merged["k2"] = rows2["k_power_2"]
    merged["k3"] = rows3["k_power_3"]
    merged["lambda_k2"] = rows2["lambda_power_2"]
    merged["lambda_k3"] = rows3["lambda_power_3"]
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot(time, pedal, corrected2, corrected3, output_path)

    print(f"Baseline source: {'fresh warm-up' if not using_saved_state else 'saved state'}")
    print(f"Saved comparison plot: {output_path.name}")
    print(f"Saved comparison CSV: {csv_output_path.name}")


if __name__ == "__main__":
    main()
