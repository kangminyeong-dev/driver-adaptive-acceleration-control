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


OUTPUT_PNG = "pedal_correction_overlay.png"
OUTPUT_CSV = "pedal_correction_output.csv"
ENVIRONMENT_ENABLE = 1.0
OUTLIER_K = 1.0


def load_event_table(csv_path, split_time):
    df = pa.load_data(csv_path)
    time = df["time_sec"].to_numpy()
    pedal = df["pedal_percent"].to_numpy()
    events = pa.detect_events(time, pedal)
    event_table = pa.build_event_table(time, pedal, events, split_time)
    return df, time, pedal, event_table


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def correction_lambda(reference_slope, current_slope, env_enable=ENVIRONMENT_ENABLE, k=OUTLIER_K):
    if current_slope is None or not np.isfinite(current_slope) or current_slope <= 0:
        return 1.0
    if reference_slope is None or not np.isfinite(reference_slope) or reference_slope <= 0:
        return 1.0

    ratio = clamp(reference_slope / current_slope, 0.0, 1.0)
    lam = 1.0 - env_enable * (1.0 - k * ratio)
    return clamp(lam, 0.0, 1.0)


def apply_event_lambda(corrected, event_row, lam):
    start_idx = int(event_row.start_idx)
    end_idx = int(event_row.end_idx)
    corrected[start_idx : end_idx + 1] *= lam


def build_corrected_series(time, pedal, event_table, split_time, saved_state=None):
    corrected = pedal.astype(float).copy()
    state = saved_state if saved_state is not None else pa.empty_baseline_state()
    using_saved_state = saved_state is not None
    decisions = []

    for row in event_table.itertuples(index=False):
        if row.slope is None or not np.isfinite(row.slope):
            continue

        if not using_saved_state and row.end_time <= split_time:
            pa.update_baseline_state(state, row.slope, 1.0)
            decisions.append(
                {
                    "event_no": row.event_no,
                    "start_time": row.start_time,
                    "end_time": row.end_time,
                    "slope": row.slope,
                    "label": "WARMUP",
                    "z": np.nan,
                    "lambda": 1.0,
                    "mean_before": np.nan,
                    "sigma_before": np.nan,
                }
            )
            continue

        should_judge = using_saved_state or row.start_time >= split_time
        if not should_judge or not pa.state_ready(state):
            continue

        mean_before = state["mean"]
        sigma_before = state["sigma"]
        z = pa.z_score(row.slope, mean_before, sigma_before)
        label = pa.classify_z(row.slope, mean_before, sigma_before)
        lam = 1.0

        if label == "OUTLIER":
            lam = correction_lambda(mean_before, row.slope)
            apply_event_lambda(corrected, row, lam)

        weight, _ = pa.baseline_update_weight(row.slope, mean_before, sigma_before)
        if weight > 0:
            pa.update_baseline_state(state, row.slope, weight)

        decisions.append(
            {
                "event_no": row.event_no,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "slope": row.slope,
                "label": label,
                "z": z,
                "lambda": lam,
                "mean_before": mean_before,
                "sigma_before": sigma_before,
            }
        )

    return corrected, pd.DataFrame(decisions), state


def save_overlay_plot(time, raw_pedal, corrected_pedal, decisions, output_path, split_time, using_saved_state):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(time, raw_pedal, color="#d62728", linewidth=2.0, label="raw pedal")
    ax.plot(time, corrected_pedal, color="#1f77b4", linewidth=2.0, label="corrected pedal")

    if not using_saved_state:
        ax.axvline(split_time, color="#555555", linestyle="--", linewidth=1.4, alpha=0.8)

    for row in decisions.itertuples(index=False):
        if row.label == "OUTLIER":
            ax.axvspan(row.start_time, row.end_time, color="#d62728", alpha=0.10, linewidth=0)

    ax.set_title("Raw vs Corrected Pedal Input")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pedal (%)")
    ax.set_ylim(-2, max(float(np.max(raw_pedal)), float(np.max(corrected_pedal)), 10.0) * 1.12)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw pedal input against OUTLIER-corrected pedal input."
    )
    parser.add_argument("--csv", default=pa.CSV_NAME, help=f"CSV path. Default: {pa.CSV_NAME}")
    parser.add_argument("--state", default=pa.STATE_NAME, help=f"Baseline state path. Default: {pa.STATE_NAME}")
    parser.add_argument("--split-time", type=float, default=pa.SPLIT_TIME_SEC, help=f"Warm-up split time. Default: {pa.SPLIT_TIME_SEC:g}")
    parser.add_argument("--output", default=OUTPUT_PNG, help=f"Output plot path. Default: {OUTPUT_PNG}")
    parser.add_argument("--csv-output", default=OUTPUT_CSV, help=f"Output corrected CSV path. Default: {OUTPUT_CSV}")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = base_dir / state_path

    df, time, pedal, event_table = load_event_table(csv_path, args.split_time)
    saved_state = pa.load_baseline_state(state_path)
    using_saved_state = saved_state is not None
    corrected, decisions, final_state = build_corrected_series(
        time,
        pedal,
        event_table,
        args.split_time,
        saved_state,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    csv_output_path = Path(args.csv_output)
    if not csv_output_path.is_absolute():
        csv_output_path = base_dir / csv_output_path

    out_df = df.copy()
    out_df["corrected_pedal_percent"] = corrected
    out_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    save_overlay_plot(time, pedal, corrected, decisions, output_path, args.split_time, using_saved_state)

    judged = decisions[decisions["label"].isin(["OK", "WARN", "OUTLIER"])]
    print(f"Baseline source: {'saved state' if using_saved_state else 'warm-up in this CSV'}")
    print(f"Judged events: {len(judged)}")
    print(f"OUTLIER corrected events: {int((decisions['label'] == 'OUTLIER').sum())}")
    print(f"Final simulated mean={final_state['mean']:.3f}, sigma={final_state['sigma']:.3f}, n={final_state['sample_count']:.2f}")
    print(f"Saved plot: {output_path.name}")
    print(f"Saved corrected CSV: {csv_output_path.name}")


if __name__ == "__main__":
    main()
