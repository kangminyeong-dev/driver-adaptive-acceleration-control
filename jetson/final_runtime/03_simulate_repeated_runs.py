import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_pedal_analysis():
    module_path = Path(__file__).resolve().parent / "02_pedal_analysis.py"
    spec = importlib.util.spec_from_file_location("pedal_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pa = load_pedal_analysis()


DEFAULT_RUNS = 20
LOG_NAME = "outputs/pedal_repeat_simulation_log.csv"
PLOT_NAME = "outputs/pedal_repeat_simulation_trend.png"


def load_event_table(csv_path, split_time):
    df = pa.load_data(csv_path)
    time = df["time_sec"].to_numpy()
    pedal = df["pedal_percent"].to_numpy()
    events = pa.detect_events(time, pedal)
    return pa.build_event_table(time, pedal, events, split_time)


def event_is_valid(row):
    return row.slope is not None and pd.notna(row.slope)


def should_judge_event(row, run_no, split_time):
    if run_no == 1:
        return row.start_time >= split_time
    return True


def should_warmup_event(row, run_no, split_time):
    return run_no == 1 and row.end_time <= split_time


def simulate_one_run(event_table, state, run_no, split_time):
    rows = []

    for row in event_table.itertuples(index=False):
        if not event_is_valid(row):
            continue

        if should_warmup_event(row, run_no, split_time):
            pa.update_baseline_state(state, row.slope, 1.0)
            rows.append(
                {
                    "run": run_no,
                    "event_no": row.event_no,
                    "phase": "warmup",
                    "start_time": row.start_time,
                    "end_time": row.end_time,
                    "slope": row.slope,
                    "z": None,
                    "label": "WARMUP",
                    "action": "learned",
                    "weight": 1.0,
                    "mean_after": state["mean"],
                    "sigma_after": state["sigma"],
                    "n_after": state["sample_count"],
                }
            )
            continue

        if not should_judge_event(row, run_no, split_time):
            continue
        if not pa.state_ready(state):
            continue

        mu = state["mean"]
        sigma = state["sigma"]
        z = pa.z_score(row.slope, mu, sigma)
        label = pa.classify_z(row.slope, mu, sigma)
        weight, action = pa.baseline_update_weight(row.slope, mu, sigma)

        if weight > 0:
            pa.update_baseline_state(state, row.slope, weight)
        else:
            pa.refresh_state_stats(state)

        rows.append(
            {
                "run": run_no,
                "event_no": row.event_no,
                "phase": "judge",
                "start_time": row.start_time,
                "end_time": row.end_time,
                "slope": row.slope,
                "z": z,
                "label": label,
                "action": action,
                "weight": weight,
                "mean_after": state["mean"],
                "sigma_after": state["sigma"],
                "n_after": state["sample_count"],
            }
        )

    return rows


def summarize_runs(log_df):
    judged = log_df[log_df["phase"] == "judge"].copy()
    if judged.empty:
        return pd.DataFrame()

    summary = (
        judged.groupby("run")
        .agg(
            judged_events=("event_no", "count"),
            ok_count=("label", lambda s: int((s == "OK").sum())),
            warn_count=("label", lambda s: int((s == "WARN").sum())),
            outlier_count=("label", lambda s: int((s == "OUTLIER").sum())),
            mean=("mean_after", "last"),
            sigma=("sigma_after", "last"),
            n=("n_after", "last"),
        )
        .reset_index()
    )
    return summary


def save_trend_plot(summary_df, output_path):
    if summary_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(summary_df["run"], summary_df["mean"], marker="o", label="mean")
    axes[0].plot(summary_df["run"], summary_df["sigma"], marker="o", label="sigma")
    axes[0].set_ylabel("Baseline value")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].bar(summary_df["run"], summary_df["outlier_count"], label="OUTLIER")
    axes[1].plot(summary_df["run"], summary_df["warn_count"], marker="o", color="#f58518", label="WARN")
    axes[1].plot(summary_df["run"], summary_df["ok_count"], marker="o", color="#1f9d55", label="OK")
    axes[1].set_xlabel("Repeated 2-minute pass")
    axes[1].set_ylabel("Judged event count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Repeated Pedal Gaussian Simulation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Fast repeated simulation of the online pedal Gaussian detector."
    )
    parser.add_argument("--csv", default=pa.CSV_NAME, help=f"CSV path. Default: {pa.CSV_NAME}")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Number of 2-minute passes. Default: {DEFAULT_RUNS}")
    parser.add_argument("--split-time", type=float, default=pa.SPLIT_TIME_SEC, help=f"Warm-up split time. Default: {pa.SPLIT_TIME_SEC:g}")
    parser.add_argument("--log", default=LOG_NAME, help=f"Per-event log CSV. Default: {LOG_NAME}")
    parser.add_argument("--plot", default=PLOT_NAME, help=f"Trend plot PNG. Default: {PLOT_NAME}")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0.")

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    event_table = load_event_table(csv_path, args.split_time)
    state = pa.empty_baseline_state()
    all_rows = []

    print(f"Loaded {len(event_table)} pedal events from {csv_path.name}")
    print(f"Run 1: 0-{args.split_time:g}s warm-up, then judge")
    print("Run 2+: judge the full 2-minute sequence")
    print()

    for run_no in range(1, args.runs + 1):
        run_rows = simulate_one_run(event_table, state, run_no, args.split_time)
        all_rows.extend(run_rows)
        judged = [row for row in run_rows if row["phase"] == "judge"]
        ok_count = sum(row["label"] == "OK" for row in judged)
        warn_count = sum(row["label"] == "WARN" for row in judged)
        outlier_count = sum(row["label"] == "OUTLIER" for row in judged)
        print(
            f"run {run_no:02d}: judged={len(judged):2d}, "
            f"OK={ok_count:2d}, WARN={warn_count:2d}, OUTLIER={outlier_count:2d}, "
            f"mean={state['mean']:.3f}, sigma={state['sigma']:.3f}, n={state['sample_count']:.2f}"
        )

    log_df = pd.DataFrame(all_rows)
    summary_df = summarize_runs(log_df)

    log_path = base_dir / args.log
    plot_path = base_dir / args.plot
    log_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    save_trend_plot(summary_df, plot_path)

    print()
    print(f"Saved event log: {log_path.name}")
    print(f"Saved trend plot: {plot_path.name}")


if __name__ == "__main__":
    main()
