"""Compact monthly heatmaps for variable-level Hellinger-CUSUM monitoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def results_to_monthly_cusum(
    results_by_variable: Mapping[str, Sequence[dict]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate monitoring results to monthly maximum normalised CUSUM.

    ``results_by_variable`` maps each variable name to the list returned by
    ``rolling_metric_fixed_baseline_cusum``. Values are normalised as C_t / h,
    so 1 represents the alert threshold regardless of variable type.
    """
    frames = []

    for variable, results in results_by_variable.items():
        frame = pd.DataFrame(results)
        if frame.empty:
            continue

        required = {"end_date", "cusum", "threshold", "alert"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{variable} is missing monitoring fields: {sorted(missing)}")

        frame = frame.copy()
        frame["Variable"] = variable
        frame["end_date"] = pd.to_datetime(frame["end_date"])
        frame["month"] = frame["end_date"].dt.to_period("M")
        frame["normalised_cusum"] = frame["cusum"] / frame["threshold"]
        frames.append(frame[["Variable", "month", "normalised_cusum", "alert"]])

    if not frames:
        raise ValueError("No monitoring results were supplied.")

    long_data = pd.concat(frames, ignore_index=True)
    variable_order = list(results_by_variable.keys())

    monthly_cusum = (
        long_data.groupby(["Variable", "month"], observed=True)["normalised_cusum"]
        .max()
        .unstack("month")
        .reindex(variable_order)
        .sort_index(axis=1)
    )

    monthly_alerts = (
        long_data.assign(alert=lambda df: df["alert"].astype(bool))
        .groupby(["Variable", "month"], observed=True)["alert"]
        .max()
        .unstack("month")
        .reindex(index=variable_order, columns=monthly_cusum.columns, fill_value=False)
        .fillna(False)
    )

    return monthly_cusum, monthly_alerts


def humanise_variable_name(variable: str) -> str:
    """Convert a dataset variable name into a readable heatmap label."""
    prefixes = ("symptoms_", "comorbid_", "comps_", "vs_")
    label = variable
    for prefix in prefixes:
        if label.startswith(prefix):
            label = label.removeprefix(prefix)
            break
    return label.replace("_", " ").replace("ards", "ARDS").title()


def plot_monthly_cusum_heatmap(
    results_by_variable: Mapping[str, Sequence[dict]],
    *,
    title: str,
    events: Mapping[str, str] | None = None,
    variable_labels: Mapping[str, str] | None = None,
    cmap_name: str = "viridis",
    figsize: tuple[float, float] = (15, 7),
):
    """Plot monthly maximum C_t/h values with a separate event timeline.

    Values from 0 to 1 use the colour scale. Values above 1 are coloured red,
    indicating a threshold crossing. Grey cells indicate months without a
    monitoring result, for example while a variable was being re-baselined.
    """
    monthly_cusum, monthly_alerts = results_to_monthly_cusum(results_by_variable)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_over("crimson")
    cmap.set_bad("#f0f0f0")
    norm = colors.Normalize(vmin=0, vmax=1, clip=False)

    data = np.ma.masked_invalid(monthly_cusum.to_numpy(dtype=float))
    # Reserve a dedicated colourbar column.  Letting ``fig.colorbar`` resize
    # only the heatmap axis would make its month cells physically narrower
    # than the shared event timeline below.
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[10, 3],
        width_ratios=[1, 0.022],
        hspace=0.08,
        wspace=0.04,
    )
    ax = fig.add_subplot(grid[0, 0])
    timeline_ax = fig.add_subplot(grid[1, 0], sharex=ax)
    colourbar_ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    months = monthly_cusum.columns
    if len(months):
        tick_step = max(1, int(np.ceil(len(months) / 12)))
        tick_positions = np.arange(0, len(months), tick_step)
        tick_labels = [months[position].strftime("%b\n%Y") for position in tick_positions]
        timeline_ax.set_xticks(tick_positions, tick_labels)

    if variable_labels is None:
        variable_labels = {}
    y_labels = [
        variable_labels.get(variable, humanise_variable_name(variable))
        for variable in monthly_cusum.index
    ]
    ax.set_yticks(np.arange(len(y_labels)), y_labels)

    if events:
        event_positions = {month: index for index, month in enumerate(months)}
        timeline_y = 0
        timeline_ax.axhline(timeline_y, color="#536878", linewidth=1.8)

        for event_index, (event_date, event_label) in enumerate(events.items()):
            event_month = pd.Timestamp(event_date).to_period("M")
            if event_month in event_positions:
                x_position = event_positions[event_month]
                label_level = 0.80 if event_index % 2 == 0 else -0.80
                timeline_ax.vlines(
                    x_position,
                    timeline_y,
                    label_level,
                    color="#1f355e",
                    linewidth=1.6,
                )
                timeline_ax.scatter(
                    x_position,
                    timeline_y,
                    s=45,
                    color="#0b7285",
                    zorder=3,
                )
                timeline_ax.annotate(
                    event_label,
                    xy=(x_position, label_level),
                    xytext=(0, 5 if label_level > 0 else -5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if label_level > 0 else "top",
                    rotation=0,
                    fontsize=11,
                    color="#1f355e",
                )

    colourbar = fig.colorbar(image, cax=colourbar_ax, extend="max")
    colourbar.set_label(r"Monthly maximum normalised CUSUM ($C_t/h$)")
    colourbar.set_ticks(np.linspace(0, 1, 6))

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Variable")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlim(-0.5, len(months) - 0.5)

    timeline_ax.set_ylim(-1.25, 1.25)
    timeline_ax.set_yticks([])
    timeline_ax.set_ylabel("Events")
    timeline_ax.set_xlabel("Calendar month", labelpad=10)
    timeline_ax.tick_params(axis="x", pad=8)
    timeline_ax.spines[["top", "right", "left"]].set_visible(False)

    fig.subplots_adjust(left=0.20, right=0.92, top=0.91, bottom=0.18)

    return fig, ax, monthly_cusum, monthly_alerts
