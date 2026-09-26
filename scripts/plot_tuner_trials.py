#!/usr/bin/env python3
"""
plot_tuner_trials.py

Plots the full per-epoch metric trajectory (default: val_loss) for the
top N trials of a completed (or in-progress) tune.py search, so you can
actually see HOW each trial's validation loss evolved -- not just its
single final score.

Why this exists: Keras Tuner's own trial.json does NOT retain a full
epoch-by-epoch history. Checked directly against this project's own
tuner_runs/: every completed trial.json holds exactly ONE observation
per metric, because Tuner.on_epoch_end() is a no-op by default (Keras
Tuner's own docstring: "Intermediate results are not passed to the
Oracle") -- the oracle only ever learns a trial's *final* result. That
single collapsed number can't distinguish "val_loss fell smoothly and
stayed down" from "val_loss dipped early, then flattened or got worse"
-- exactly the distinction that separates a genuinely good learning
rate from one that's only transiently good (see the comment on
tuner_search_space.learning_rate in config.yaml, and the discussion
that led here).

tune.py now has a _LogEpochHistory mixin that overrides
Tuner.on_epoch_end() (the real per-epoch hook Keras Tuner already calls
via TunerCallback, previously discarded) to write each epoch's train+val
metrics to <trial_dir>/epoch_history.csv as training happens. This
script reads those files.

IMPORTANT: only trials run AFTER that tune.py change will have an
epoch_history.csv. Trials from before it (including anything already
sitting in tuner_runs/ from earlier searches) only have the single
collapsed value trial.json always had -- this script reports that
plainly per trial rather than pretending to have data it doesn't, so
don't be surprised if your currently-running/older search shows "no
epoch history recorded" for every trial. Let the search (re)run for a
bit with the updated tune.py and this will start filling in.

Requires:
    pip install matplotlib   # optional -- CSV summary + console report
                              # work without it; only the plot needs it.

Usage (everything from config.yaml):
    python scripts/plot_tuner_trials.py

Usage (custom):
    python scripts/plot_tuner_trials.py --top-n 8 --metric val_perplexity
"""

import argparse
import csv
import json
from pathlib import Path

from config import load_config


def parse_args(cfg):
    parser = argparse.ArgumentParser(
        description="Plot per-epoch metric trajectories for the top trials of a tune.py search."
    )
    parser.add_argument("--project-dir", type=Path, default=cfg["paths"]["tuner_project_dir"],
                         help="Where Keras Tuner stores trial results (same as tune.py's --project-dir).")
    parser.add_argument("--project-name", default=cfg["tuner_run"]["project_name"],
                         help="Same as tune.py's --project-name.")
    parser.add_argument("--top-n", type=int, default=5,
                         help="How many of the best-scoring COMPLETED trials to plot. Default: 5.")
    parser.add_argument("--metric", default="val_loss",
                         help="Which logged metric to plot on the y-axis (must be a column tune.py's "
                              "epoch_history.csv would contain, e.g. val_loss, val_perplexity, "
                              "val_top1_acc). Default: val_loss.")
    parser.add_argument("--maximize-score", action="store_true",
                         help="Rank trials by highest trial.json 'score' instead of lowest. Only needed "
                              "if tune.py's objective direction was changed from this project's default "
                              "(val_loss, minimize).")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to write results.png / trial_summary.csv. Default: "
                              "<project-dir>/<project-name>_trial_plots/")
    return parser.parse_args()


def load_completed_trials(project_dir, project_name):
    """Return one dict per COMPLETED trial: trial_id, score, best_step,
    hyperparameters (values dict), and its directory Path.
    """
    base = Path(project_dir) / project_name
    trials = []
    for trial_dir in sorted(base.glob("trial_*")):
        trial_json = trial_dir / "trial.json"
        if not trial_json.exists():
            continue
        with trial_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") != "COMPLETED" or data.get("score") is None:
            continue
        trials.append({
            "trial_id": data["trial_id"],
            "score": data["score"],
            "best_step": data.get("best_step"),
            "hyperparameters": data.get("hyperparameters", {}).get("values", {}),
            "dir": trial_dir,
        })
    return trials


def load_epoch_history(trial_dir):
    """Return a list of {column: str_value} row dicts from
    <trial_dir>/epoch_history.csv, or None if that file doesn't exist
    (trial predates the _LogEpochHistory mixin in tune.py).
    """
    history_path = trial_dir / "epoch_history.csv"
    if not history_path.exists():
        return None
    with history_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def series_for(rows, metric):
    """Extract (epoch, value) pairs for `metric` from history rows,
    skipping any row missing that column (shouldn't normally happen,
    but metrics could differ if the model/config changed mid-search).
    """
    points = []
    for row in rows:
        raw = row.get(metric)
        if raw in (None, ""):
            continue
        try:
            points.append((int(float(row["epoch"])), float(raw)))
        except (TypeError, ValueError):
            continue
    points.sort(key=lambda p: p[0])
    return points


def describe_trajectory(points):
    """Plain-language readout of whether a trial's metric trajectory
    looks stable (monotonic improvement) or shows the "quick dip then
    flattens/worsens" pattern that indicates an unstable learning rate.
    Assumes lower is better (loss/perplexity); flip the comparison
    yourself when reading an accuracy metric.
    """
    if len(points) < 2:
        return "only one epoch recorded -- not enough to judge a trend"
    values = [v for _, v in points]
    increases = [(i, points[i][0]) for i in range(1, len(values)) if values[i] > values[i - 1]]
    if not increases:
        return f"monotonic improvement across all {len(values)} recorded epochs -- looks stable"
    first_idx, first_epoch = increases[0]
    return (
        f"improved for {first_idx} epoch(s), then got worse at epoch {first_epoch} "
        f"({values[first_idx - 1]:.4f} -> {values[first_idx]:.4f}); {len(increases)} "
        f"worsening step(s) total out of {len(values) - 1} -- possible instability, not just noise "
        f"if this keeps happening at the same point across several top trials"
    )


def companion_metric(metric):
    """val_loss <-> loss, val_perplexity <-> perplexity, etc."""
    if metric.startswith("val_"):
        return metric[len("val_"):]
    return f"val_{metric}"


def write_summary_csv(trials_with_data, metric, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "score", "learning_rate", "epochs_recorded", "trajectory"])
        for t in trials_with_data:
            hp = t["hyperparameters"]
            writer.writerow([
                t["trial_id"],
                t["score"],
                hp.get("learning_rate", ""),
                len(t["points"]),
                t["trajectory_note"],
            ])


def write_plot(trials_with_data, metric, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n(matplotlib not installed -- skipping {path.name}; "
              f"`pip install matplotlib` to get the plot.)")
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    companion = companion_metric(metric)
    for t in trials_with_data:
        color = None
        label = f"trial_{t['trial_id']} (lr={t['hyperparameters'].get('learning_rate', '?'):.2e})" \
            if isinstance(t["hyperparameters"].get("learning_rate"), float) \
            else f"trial_{t['trial_id']}"
        epochs = [e for e, _ in t["points"]]
        values = [v for _, v in t["points"]]
        line, = ax.plot(epochs, values, marker="o", label=label)
        color = line.get_color()

        companion_points = series_for(t["rows"], companion)
        if companion_points:
            c_epochs = [e for e, _ in companion_points]
            c_values = [v for _, v in companion_points]
            ax.plot(c_epochs, c_values, linestyle="--", alpha=0.5, color=color)

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Top trials: {metric} (solid) vs {companion_metric(metric)} (dashed) per epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    cfg = load_config()
    args = parse_args(cfg)

    output_dir = args.output_dir or (Path(args.project_dir) / f"{args.project_name}_trial_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trials = load_completed_trials(args.project_dir, args.project_name)
    if not all_trials:
        print(f"No COMPLETED trials found under {Path(args.project_dir) / args.project_name}. "
              f"Has tune.py finished at least one trial?")
        return

    all_trials.sort(key=lambda t: t["score"], reverse=args.maximize_score)
    top_trials = all_trials[: args.top_n]

    print(f"Found {len(all_trials)} completed trial(s); reporting on top {len(top_trials)} "
          f"by trial.json score ({'highest' if args.maximize_score else 'lowest'} first):\n")

    trials_with_data = []
    trials_without_data = []
    for t in top_trials:
        rows = load_epoch_history(t["dir"])
        if rows is None:
            trials_without_data.append(t)
            continue
        points = series_for(rows, args.metric)
        if not points:
            print(f"  trial_{t['trial_id']}: epoch_history.csv exists but has no '{args.metric}' "
                  f"column -- check --metric spelling against the CSV's header.")
            continue
        t["rows"] = rows
        t["points"] = points
        t["trajectory_note"] = describe_trajectory(points)
        trials_with_data.append(t)
        print(f"  trial_{t['trial_id']} (score={t['score']:.4f}): {t['trajectory_note']}")

    if trials_without_data:
        ids = ", ".join(f"trial_{t['trial_id']}" for t in trials_without_data)
        print(f"\n  No epoch_history.csv for: {ids} -- these ran before tune.py's "
              f"_LogEpochHistory mixin was added, so only their single final score survives. "
              f"Let the search run a bit further with the updated tune.py to get real "
              f"per-epoch trajectories for trials like these.")

    if not trials_with_data:
        print("\nNothing to plot yet -- no top trial has per-epoch history recorded. "
              "Run/continue tune.py with the updated script, then re-run this.")
        return

    plot_path = output_dir / f"top_trials_{args.metric}.png"
    plotted = write_plot(trials_with_data, args.metric, plot_path)

    summary_path = output_dir / "top_trials_summary.csv"
    write_summary_csv(trials_with_data, args.metric, summary_path)

    print(f"\nSummary CSV: {summary_path}")
    if plotted:
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
