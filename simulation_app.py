# app.py
# A simple Plotly Dash UI to tweak generate_scenario() params and generate a dataframe + preview + download.
#
# Run:
#   pip install dash pandas numpy plotly scipy
#   python app.py

from __future__ import annotations

import json
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import distance
from scipy import stats


# -----------------------------
# Minimal data-generation helpers (replace with your canonical ones if you prefer)
# -----------------------------
def _sample_admission_dates(
    n: int,
    start_date: str,
    end_date: str,
    pattern: str = "spiky",
    seed: int = 0,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = pd.date_range(start, end, freq="D")
    if len(days) == 0:
        raise ValueError("Empty date range.")

    if pattern.lower() == "uniform":
        w = np.ones(len(days), dtype=float)
    else:
        dow = np.array([d.dayofweek for d in days])  # 0=Mon
        w = 0.6 + 0.4 * (dow == 0) + 0.2 * (dow == 1)  # Mon/Tue higher
        if len(days) >= 10:
            spike_idx = rng.choice(len(days), size=min(3, len(days)), replace=False)
            w[spike_idx] *= 2.5

    w = w / w.sum()
    chosen = rng.choice(days, size=int(n), replace=True, p=w)
    return pd.to_datetime(chosen).sort_values().to_series(index=None)


def continuous_column(
    n_patients: int,
    start_date: str,
    end_date: str,
    dist: str = "normal",
    low: float = 0,
    high: float = 95,
    mean: float = 60,
    sd: float = 25,
    col_integer: bool = True,
    col_decimals: int = 1,
    admissions_pattern: str = "spiky",
    seed: int = 0,
    col_name: str = "var",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _sample_admission_dates(n_patients, start_date, end_date, admissions_pattern, seed=seed)

    dist = dist.lower()
    if dist == "normal":
        vals = rng.normal(loc=mean, scale=sd, size=n_patients)
    elif dist == "uniform":
        vals = rng.uniform(low=low, high=high, size=n_patients)
    else:
        raise ValueError("dist must be 'normal' or 'uniform' (in this minimal helper).")

    vals = np.clip(vals, low, high)

    if col_integer:
        vals = np.rint(vals).astype(int)
    else:
        vals = np.round(vals.astype(float), col_decimals)

    return pd.DataFrame({"date_admit": dates.values, col_name: vals})


def binary_column(
    n_patients: int,
    start_date: str,
    end_date: str,
    cough_p: float = 0.3,
    admissions_pattern: str = "spiky",
    seed: int = 0,
    col_name: str = "var",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _sample_admission_dates(n_patients, start_date, end_date, admissions_pattern, seed=seed)
    p = float(np.clip(cough_p, 0.0, 1.0))
    vals = rng.binomial(n=1, p=p, size=n_patients).astype(int)
    return pd.DataFrame({"date_admit": dates.values, col_name: vals})


# -----------------------------
# generate_scenario()
# -----------------------------
def generate_scenario(
    kind: str,
    scenario_type: str,
    start_date: str,
    end_date: str,
    events: dict | None,
    n_per_block: int = 400,
    n_per_block_list: list[int] | None = None,
    admissions_pattern: str = "spiky",
    seed_start: int = 100,
    col_name: str = "var",
    values: list[float] | None = None,
    sds: list[float] | None = None,
    dist: str = "normal",
    low: float = 0,
    high: float = 95,
    mean: float = 60,
    sd: float = 25,
    col_integer: bool = True,
    col_decimals: int = 1,
    p: float = 0.30,
):
    kind = kind.lower()
    scenario_type = scenario_type.lower()
    if kind not in {"continuous", "binary"}:
        raise ValueError("kind must be 'continuous' or 'binary'")
    if scenario_type not in {"gradual_drift", "step_change", "transient_peak", "high_noise"}:
        raise ValueError("scenario_type must be one of: gradual_drift, step_change, transient_peak, high_noise")

    if events is None:
        event_dates = []
    else:
        event_dates = sorted(pd.to_datetime(list(events.keys())))

    boundaries = [pd.to_datetime(start_date)] + event_dates + [pd.to_datetime(end_date)]

    blocks = []
    for i in range(len(boundaries) - 1):
        b_start = boundaries[i]
        if i < len(boundaries) - 2:
            b_end = boundaries[i + 1] - pd.Timedelta(days=1)
        else:
            b_end = boundaries[i + 1]
        blocks.append((str(b_start.date()), str(b_end.date())))

    n_blocks = len(blocks)

    if n_per_block_list is not None:
        if len(n_per_block_list) != n_blocks:
            raise ValueError(f"`n_per_block_list` length ({len(n_per_block_list)}) must match number of blocks ({n_blocks}).")
        ns = list(n_per_block_list)
    else:
        ns = [int(n_per_block)] * n_blocks

    if scenario_type in {"gradual_drift", "step_change", "transient_peak"} and values is None:
        if kind == "continuous":
            if scenario_type == "gradual_drift":
                end_mean = mean - 20
                values = [mean + (end_mean - mean) * i / (n_blocks - 1) for i in range(n_blocks)] if n_blocks > 1 else [mean]
            elif scenario_type == "step_change":
                values = [mean, mean - 20]
            elif scenario_type == "transient_peak":
                values = [mean, mean - 30, mean]
        else:
            if scenario_type == "gradual_drift":
                end_p = min(0.95, p + 0.20)
                values = [p + (end_p - p) * i / (n_blocks - 1) for i in range(n_blocks)] if n_blocks > 1 else [p]
            elif scenario_type == "step_change":
                values = [p, min(0.95, p + 0.20)]
            elif scenario_type == "transient_peak":
                values = [p, min(0.95, p + 0.35), p]

    if scenario_type == "high_noise":
        if kind == "continuous":
            if values is None:
                values = [mean] * n_blocks
            if sds is None:
                if n_blocks == 1:
                    sds = [sd]
                else:
                    split = n_blocks // 2
                    sds = [sd] * split + [sd * 2] * (n_blocks - split)
        else:
            if values is None:
                values = [p] * n_blocks

    if values is None:
        raise ValueError("values schedule could not be determined; please provide `values`.")

    if len(values) != n_blocks:
        raise ValueError(f"`values` length ({len(values)}) must match number of blocks ({n_blocks}).")

    if kind == "continuous" and sds is not None and len(sds) != n_blocks:
        raise ValueError(f"`sds` length ({len(sds)}) must match number of blocks ({n_blocks}).")

    dfs = []
    for i, ((start, end), v, n_i) in enumerate(zip(blocks, values, ns)):
        seed_i = seed_start + i

        if kind == "continuous":
            sd_i = sds[i] if sds is not None else sd
            dfs.append(
                continuous_column(
                    n_patients=int(n_i),
                    start_date=start,
                    end_date=end,
                    dist=dist,
                    low=low,
                    high=high,
                    mean=float(v),
                    sd=float(sd_i),
                    col_integer=col_integer,
                    col_decimals=col_decimals,
                    admissions_pattern=admissions_pattern,
                    seed=seed_i,
                    col_name=col_name,
                )
            )
        else:
            dfs.append(
                binary_column(
                    n_patients=int(n_i),
                    start_date=start,
                    end_date=end,
                    cough_p=float(v),
                    admissions_pattern=admissions_pattern,
                    seed=seed_i,
                    col_name=col_name,
                )
            )

    return pd.concat(dfs, ignore_index=True).sort_values("date_admit").reset_index(drop=True)


# -----------------------------
# Plot helpers
# -----------------------------
def fig_iqr(arr):
    arr = np.asarray(arr, dtype=float)
    median = arr[:, 0]
    q1 = arr[:, 1]
    q3 = arr[:, 2]

    x = np.arange(len(arr))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=median,
            mode="markers",
            marker=dict(size=8),
            error_y=dict(
                type="data",
                symmetric=False,
                array=q3 - median,
                arrayminus=median - q1,
                thickness=1.5,
                width=4
            ),
            name="Median (Q1–Q3)"
        )
    )
    fig.update_layout(template="simple_white", xaxis_title="Item", yaxis_title="Value", showlegend=False, height=450)
    return fig


def fig_percentage_bar(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] != 2:
        raise ValueError("arr must be of shape (n, 2): [count, percentage]")

    counts = arr[:, 0]
    perc = arr[:, 1]
    complement = 100 - perc
    x = np.arange(1, len(arr) + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=perc,
            name="Percentage",
            customdata=np.column_stack((counts, perc)),
            hovertemplate="Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1f}%<extra></extra>"
        )
    )
    fig.add_trace(go.Bar(x=x, y=complement, name="Remaining", hoverinfo="skip"))
    fig.update_layout(barmode="stack", template="simple_white", yaxis=dict(range=[0, 100]))
    return fig


# -----------------------------
# Metrics
# -----------------------------
def HellingerDistance(a, b, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    if not np.isin(a, [0.0, 1.0]).all() or not np.isin(b, [0.0, 1.0]).all():
        raise ValueError("Inputs must be binary (0/1/NaN).")
    p1 = a.mean()
    p2 = b.mean()
    h = (1 / np.sqrt(2)) * np.sqrt((np.sqrt(p1) - np.sqrt(p2)) ** 2 + (np.sqrt(1 - p1) - np.sqrt(1 - p2)) ** 2)
    return float(h)


def CohenH(a, b, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    p1 = a.mean()
    p2 = b.mean()
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return float(np.abs(h))


def prevalenceChange(a, b, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    return float(abs(a.mean() - b.mean()))


def FisherExactTest(a, b, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    table = np.array([[np.sum(a == 1), np.sum(a == 0)], [np.sum(b == 1), np.sum(b == 0)]])
    _, p_value = stats.fisher_exact(table)
    return float(p_value)


def JSDivergence(a, b, n_bins=10, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Inputs must contain at least one non-NaN value each.")
    pooled = np.concatenate([a, b])
    cutoffs = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    if cutoffs.size < 2:
        return 0.0
    bins_a = pd.cut(a, bins=cutoffs, include_lowest=True, duplicates="drop")
    bins_b = pd.cut(b, bins=cutoffs, include_lowest=True, duplicates="drop")
    all_bins = pd.Categorical(pd.cut(pooled, bins=cutoffs, include_lowest=True, duplicates="drop")).categories
    count_a = pd.value_counts(bins_a, sort=False).reindex(all_bins, fill_value=0)
    count_b = pd.value_counts(bins_b, sort=False).reindex(all_bins, fill_value=0)
    p = (count_a / count_a.sum()).to_numpy(dtype=float)
    q = (count_b / count_b.sum()).to_numpy(dtype=float)
    return float(distance.jensenshannon(p, q))


def MedianShift(a, b, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    med_a = np.median(a)
    med_b = np.median(b)
    return float(abs(med_a - med_b) / med_b)


def IQRShift(a, b, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    q1_a, q3_a = np.percentile(a, [25, 75])
    q1_b, q3_b = np.percentile(b, [25, 75])
    iqr_a = q3_a - q1_a
    iqr_b = q3_b - q1_b
    return float(abs(iqr_a - iqr_b) / iqr_b)


def KStest(a, b, dropna=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    ks_stat = stats.kstest(a, b)
    return ks_stat.statistic


def binary_metrics(a, b):
    return {
        "hellinger_distance": HellingerDistance(a, b),
        "cohen_h": CohenH(a, b),
        "prevalence_change": prevalenceChange(a, b),
        "fisher_exact": FisherExactTest(a, b),
    }


def continuous_metrics(a, b, n_bins=10):
    return {
        "js_divergence": JSDivergence(a, b, n_bins=n_bins),
        "median_shift": MedianShift(a, b),
        "iqr_shift": IQRShift(a, b),
        "ks_test": KStest(a, b),
    }


def prepare_data(data, value_col, date_col, batch):
    df = data[[value_col, date_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    x = df[value_col].to_numpy(dtype=float)
    binary = set(pd.Series(x).dropna().unique()).issubset({0.0, 1.0})

    n = len(x)
    if n < 2 * batch:
        raise ValueError("Need at least 2 batches to compare.")

    freq = 'W-MON'
    df["week"] = df[date_col].dt.to_period(freq)
    weeks = np.sort(df["week"].dropna().unique())
    if len(weeks) < 2:
        raise ValueError("Need data spanning at least 2 distinct weeks.")
    return df, x, n, binary, weeks


def rolling_metric_fixed_baseline_cusum(
    data,
    value_col,
    date_col,
    events,
    batch=100,
    baseline_batches=3,
    n_bins=10,
    k=0.35,
    metric_name=None,
    plot=True,
    th=1.0,
    stable_patience=3,
    subsample_rate_low=0.2
):
    df, x, n, binary, weeks = prepare_data(data, value_col, date_col, batch)

    baseline_n = baseline_batches * batch
    if n < baseline_n + batch:
        raise ValueError(f"Not enough data for baseline ({baseline_n}) + at least 1 batch ({batch}). Got n={n}.")

    baseline = x[:baseline_n]
    results = []
    cusum = 0.0

    n_after = n - baseline_n
    n_batches = n_after // batch

    chosen_metric = None
    sampling_rate = 1.0
    rebaseline = False
    rebaseline_buf = []

    for i in range(n_batches):
        start = baseline_n + i * batch
        end = start + batch

        ref = baseline
        cur = x[start:end]

        rng = np.random.default_rng(182)
        if sampling_rate < 1.0:
            n_cur = len(cur)
            keep_n = max(1, int(np.floor(n_cur * sampling_rate)))
            idx = rng.choice(n_cur, size=keep_n, replace=False)
            cur = cur[idx]

        if rebaseline:
            rebaseline_buf.append(cur)
            if len(rebaseline_buf) < baseline_batches:
                continue
            baseline = np.concatenate(rebaseline_buf)
            cusum = 0.0
            rebaseline = False
            rebaseline_buf = []
            continue

        cur_dates = df[date_col].iloc[start:end]

        if binary:
            metrics = binary_metrics(ref, cur)
            kind = "binary"
            title = "Fixed Baseline Drift (Binary)"
            n_pos = np.nansum(cur)
            evidence = [int(n_pos), (n_pos / len(cur) * 100) if len(cur) > 0 else None]
        else:
            metrics = continuous_metrics(ref, cur, n_bins=n_bins)
            kind = "continuous"
            title = "Fixed Baseline Drift (Continuous)"
            med = np.nanmedian(cur)
            q1 = np.nanpercentile(cur, 25)
            q3 = np.nanpercentile(cur, 75)
            evidence = [med, q1, q3]

        metric_keys = list(metrics.keys())
        if chosen_metric is None:
            chosen_metric = metric_name if metric_name is not None else metric_keys[0]
            if chosen_metric not in metrics:
                raise ValueError(f"metric_name='{chosen_metric}' not found. Available: {metric_keys}")

        metric_value = metrics.get(chosen_metric, np.nan)

        excess = metric_value - k if (metric_value is not None and not np.isnan(metric_value)) else np.nan
        if not np.isnan(excess):
            cusum = max(0.0, cusum + excess)

        prev_cusum = results[-1]["cusum"] if results else 0.0
        in_alert = int(cusum >= th)
        alert = int((prev_cusum < th) and (cusum >= th))

        row = {
            "step": i + 1,
            "baseline_size": len(ref),
            "cur_size": len(cur),
            "cur_start": start,
            "cur_end": end,
            "start_date": cur_dates.min(),
            "end_date": cur_dates.max(),
            "type": kind,
            "evidence": evidence,
            "metric_used": chosen_metric,
            "metric_value": metric_value,
            "k": k,
            "excess": excess,
            "cusum": cusum,
            "alert": alert,
            "in_alert": in_alert,
            "threshold": th
        }
        row.update(metrics)
        results.append(row)

        if in_alert == 1:
            rebaseline = True
            sampling_rate = 1.0
            rebaseline_buf = []
        elif len(results) >= stable_patience and all(r["in_alert"] == 0 for r in results[-stable_patience:]):
            rebaseline = False
            sampling_rate = subsample_rate_low
        else:
            rebaseline = False
            sampling_rate = 1.0

    if plot:
        combo = batch_plots(results, title, kind, events, chosen_metric)
    else:
        combo = []

    return results, combo


def batch_plots(results, title, kind, events, metric):
    df = pd.DataFrame(results).sort_values("step").reset_index(drop=True)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["mid_date"] = df[["start_date", "end_date"]].mean(axis=1)

    evid = df["evidence"].tolist()
    fig_evi = fig_percentage_bar(evid) if kind == "binary" else fig_iqr(np.asarray(evid, dtype=float))

    fig_dist = px.line(
        df, x="mid_date", y=metric, title=title,
        labels={metric: kind.title() + " Distance", "mid_date": "Batch midpoint date"},
        range_y=[0, 1]
    )

    fig_dates = px.scatter(df, x="mid_date", y="start_date", title="", labels={"mid_date": "", "start_date": ""})
    fig_dates.update_traces(mode="lines+markers")
    for tr in fig_evi.data:
        tr.x = df["mid_date"]

    fig_cusum = px.line(df, x="mid_date", y="cusum", title="", labels={"cusum": "cusum"})
    fig_alert = px.line(df, x="mid_date", y="threshold", title="", labels={"threshold": "threshold"})
    fig_alert.update_traces(line=dict(color="red"))

    combo = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    for tr in fig_dist.data:
        combo.add_trace(tr, row=1, col=1)
    for tr in fig_dates.data:
        combo.add_trace(tr, row=2, col=1)
    for tr in fig_evi.data:
        combo.add_trace(tr, row=3, col=1)
    for tr in fig_cusum.data:
        combo.add_trace(tr, row=4, col=1)
    for tr in fig_alert.data:
        combo.add_trace(tr, row=4, col=1)

    combo.update_yaxes(title_text="Distance", range=[0, 1], row=1, col=1)
    combo.update_yaxes(title_text="Start date", row=2, col=1)
    combo.update_yaxes(autorange="reversed", row=2, col=1)
    combo.update_xaxes(title_text="Batch midpoint date", row=4, col=1)

    combo.update_layout(template="simple_white", height=750, showlegend=False, margin=dict(t=40, b=40, l=50, r=20))
    combo.update_layout(barmode="stack")

    if events:
        events_df = pd.DataFrame([(pd.to_datetime(k), v) for k, v in events.items()], columns=["event_date", "label"])
        event_points = []
        for _, r in events_df.iterrows():
            idx = (df["mid_date"] - r["event_date"]).abs().idxmin()
            event_points.append({"mid_date": df.loc[idx, "mid_date"], "start_date": df.loc[idx, "start_date"], "label": r["label"]})
        event_df = pd.DataFrame(event_points)
        combo.add_trace(
            go.Scatter(
                x=event_df["mid_date"],
                y=event_df["start_date"],
                mode="markers+text",
                text=event_df["label"],
                textposition="bottom center",
                marker=dict(symbol="diamond", size=12, color="red"),
                name="Events"
            ),
            row=2, col=1
        )

    return combo


# -----------------------------
# Dash app
# -----------------------------
DEFAULT_EVENTS = {"2021-03-13": "Change", "2021-06-01": "Change"}

# Fixed internal defaults (user no longer edits these in UI)
DEFAULT_COL_NAME = "age"          # <-- fixed variable name for the metric function
DEFAULT_SEED_START = 100

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "24px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Scenario generator + rolling drift"),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginBottom": "6px"},
            children=[
                html.Div([html.Div("Fixed value_col", style={"fontWeight": "bold"}), html.Div(DEFAULT_COL_NAME, style={"fontFamily": "monospace"})]),
                html.Div([html.Div("Fixed seed_start", style={"fontWeight": "bold"}), html.Div(str(DEFAULT_SEED_START), style={"fontFamily": "monospace"})]),
                html.Div([]),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("kind"),
                    dcc.Dropdown(
                        id="kind",
                        options=[{"label": "continuous", "value": "continuous"}, {"label": "binary", "value": "binary"}],
                        value="continuous",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("scenario_type"),
                    dcc.Dropdown(
                        id="scenario_type",
                        options=[
                            {"label": "gradual_drift", "value": "gradual_drift"},
                            {"label": "step_change", "value": "step_change"},
                            {"label": "transient_peak", "value": "transient_peak"},
                            {"label": "high_noise", "value": "high_noise"},
                        ],
                        value="gradual_drift",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("admissions_pattern"),
                    dcc.Dropdown(
                        id="admissions_pattern",
                        options=[{"label": "spiky", "value": "spiky"}, {"label": "uniform", "value": "uniform"}],
                        value="spiky",
                        clearable=False,
                    ),
                ]),
            ],
        ),

        html.Div(style={"height": "10px"}),

        html.Div(
            children=[
                html.Label("date range"),
                dcc.DatePickerRange(
                    id="date_range",
                    start_date="2021-01-01",
                    end_date="2021-09-30",
                    display_format="YYYY-MM-DD",
                    minimum_nights=0,
                ),
            ]
        ),

        html.Div(style={"height": "10px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("events (JSON dict: {\"YYYY-MM-DD\":\"Label\", ...})"),
                    dcc.Textarea(
                        id="events_json",
                        value=json.dumps(DEFAULT_EVENTS, indent=2),
                        style={"width": "100%", "height": "110px", "fontFamily": "monospace"},
                    ),
                ]),
                html.Div([
                    html.Label("n_per_block (used if n_per_block_list empty)"),
                    dcc.Input(id="n_per_block", type="number", value=400, style={"width": "100%"}),
                    html.Div(style={"height": "8px"}),
                    html.Label("n_per_block_list (comma-separated; must match #blocks)"),
                    dcc.Input(id="n_per_block_list", type="text", value="", placeholder="e.g. 100,200,150", style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("values schedule (comma-separated; optional)"),
                    dcc.Input(id="values_list", type="text", value="", placeholder="e.g. 60,55,50", style={"width": "100%"}),
                    html.Div(style={"height": "8px"}),
                    html.Label("sds schedule (comma-separated; optional; continuous only)"),
                    dcc.Input(id="sds_list", type="text", value="", placeholder="e.g. 10,20,20", style={"width": "100%"}),
                ]),
            ],
        ),

        html.Hr(),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "12px"},
            children=[
                html.Div([
                    html.H4("Continuous defaults"),
                    html.Label("dist"),
                    dcc.Dropdown(id="dist", options=[{"label": "normal", "value": "normal"}, {"label": "uniform", "value": "uniform"}],
                                 value="normal", clearable=False),
                    html.Div(style={"height": "6px"}),
                    html.Label("low"),
                    dcc.Input(id="low", type="number", value=0, style={"width": "100%"}),
                    html.Div(style={"height": "6px"}),
                    html.Label("high"),
                    dcc.Input(id="high", type="number", value=95, style={"width": "100%"}),
                    html.Div(style={"height": "6px"}),
                    html.Label("mean"),
                    dcc.Input(id="mean", type="number", value=60, style={"width": "100%"}),
                    html.Div(style={"height": "6px"}),
                    html.Label("sd"),
                    dcc.Input(id="sd", type="number", value=25, style={"width": "100%"}),
                ]),
                html.Div([
                    html.H4("Binary defaults"),
                    html.Label("p"),
                    dcc.Input(id="p", type="number", value=0.30, step=0.01, style={"width": "100%"}),
                ]),
                html.Div([
                    html.H4("Rolling drift controls"),
                    html.Label("batch"),
                    dcc.Input(id="drift_batch", type="number", value=100, min=10, step=10, style={"width": "100%"}),
                    html.Div(style={"height": "6px"}),
                    html.Label("k"),
                    dcc.Input(id="drift_k", type="number", value=0.35, step=0.01, style={"width": "100%"}),
                    html.Div(style={"height": "6px"}),
                    html.Label("th"),
                    dcc.Input(id="drift_th", type="number", value=1.0, step=0.1, style={"width": "100%"}),
                    html.Div(style={"height": "6px"}),
                    html.Label("subsample_rate_low"),
                    dcc.Input(id="drift_subsample", type="number", value=0.2, min=0.05, max=1.0, step=0.05, style={"width": "100%"}),
                ]),
                html.Div([
                    html.H4("Actions"),
                    html.Button("Generate dataframe", id="btn_generate", n_clicks=0, style={"width": "100%", "height": "44px"}),
                    html.Div(style={"height": "10px"}),
                    dcc.Download(id="download_df"),
                    html.Button("Download CSV", id="btn_download", n_clicks=0, style={"width": "100%", "height": "44px"}),
                    html.Div(style={"height": "10px"}),
                    html.Div(id="status", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"}),
                ]),
            ],
        ),

        html.Hr(),

        dcc.Store(id="df_store"),
        dcc.Store(id="events_store"),

        html.H3("Preview plot"),
        dcc.Graph(id="preview_plot"),

        html.H3("Dataframe (first 300 rows)"),
        dash_table.DataTable(
            id="df_table",
            page_size=25,
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "Arial", "fontSize": 12, "padding": "6px"},
            style_header={"fontWeight": "bold"},
        ),
    ],
)


def _parse_csv_floats(s: str) -> Optional[List[float]]:
    s = (s or "").strip()
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_csv_ints(s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    if not s:
        return None
    return [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]


def _safe_parse_events(s: str) -> Optional[Dict[str, str]]:
    s = (s or "").strip()
    if not s:
        return None
    obj = json.loads(s)
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ValueError("events must be a JSON object/dict.")
    for k, v in obj.items():
        pd.to_datetime(k)
        if not isinstance(v, str):
            obj[k] = str(v)
    return obj


@app.callback(
    Output("df_store", "data"),
    Output("events_store", "data"),
    Output("status", "children"),
    Input("btn_generate", "n_clicks"),
    State("kind", "value"),
    State("scenario_type", "value"),
    State("date_range", "start_date"),
    State("date_range", "end_date"),
    State("events_json", "value"),
    State("n_per_block", "value"),
    State("n_per_block_list", "value"),
    State("admissions_pattern", "value"),
    State("values_list", "value"),
    State("sds_list", "value"),
    State("dist", "value"),
    State("low", "value"),
    State("high", "value"),
    State("mean", "value"),
    State("sd", "value"),
    State("p", "value"),
    prevent_initial_call=True,
)
def on_generate(
    n_clicks,
    kind, scenario_type, start_date, end_date, events_json,
    n_per_block, n_per_block_list,
    admissions_pattern,
    values_list, sds_list,
    dist, low, high, mean, sd,
    p,
):
    try:
        if start_date is None or end_date is None:
            raise ValueError("Please select both start_date and end_date in the calendar.")

        events = _safe_parse_events(events_json)
        values = _parse_csv_floats(values_list)
        sds_parsed = _parse_csv_floats(sds_list)
        n_list = _parse_csv_ints(n_per_block_list)

        df = generate_scenario(
            kind=kind,
            scenario_type=scenario_type,
            start_date=str(pd.to_datetime(start_date).date()),
            end_date=str(pd.to_datetime(end_date).date()),
            events=events,
            n_per_block=int(n_per_block) if n_per_block is not None else 400,
            n_per_block_list=n_list,
            admissions_pattern=admissions_pattern,
            seed_start=int(DEFAULT_SEED_START),
            col_name=str(DEFAULT_COL_NAME),
            values=values,
            sds=sds_parsed,
            dist=str(dist),
            low=float(low),
            high=float(high),
            mean=float(mean),
            sd=float(sd),
            col_integer=True,
            col_decimals=1,
            p=float(p),
        )

        df_json = df.to_json(date_format="iso", orient="split")
        msg = (
            f"Generated: {len(df):,} rows | date range: "
            f"{df['date_admit'].min().date()} → {df['date_admit'].max().date()} | "
            f"blocks from events: {0 if events is None else len(events)}"
        )

        return df_json, events, msg

    except Exception as e:
        return no_update, no_update, f"ERROR:\n{type(e).__name__}: {e}"


@app.callback(
    Output("df_table", "data"),
    Output("df_table", "columns"),
    Output("preview_plot", "figure"),
    Input("df_store", "data"),
    Input("events_store", "data"),
    Input("drift_batch", "value"),
    Input("drift_k", "value"),
    Input("drift_th", "value"),
    Input("drift_subsample", "value"),
    prevent_initial_call=True,
)
def update_outputs(df_json, events, batch, k, th, subsample_rate_low):
    if not df_json:
        return [], [], px.scatter(title="No data yet")

    df = pd.read_json(df_json, orient="split")
    df["date_admit"] = pd.to_datetime(df["date_admit"])

    preview = df.head(300)
    cols = [{"name": c, "id": c} for c in preview.columns]
    data = preview.to_dict("records")

    # Rolling drift using UI parameters; everything else fixed as requested
    results, fig = rolling_metric_fixed_baseline_cusum(
        data=df,
        value_col=DEFAULT_COL_NAME,
        date_col="date_admit",
        events=events,
        batch=int(batch) if batch else 100,
        baseline_batches=3,
        n_bins=10,
        k=float(k) if k is not None else 0.35,
        metric_name=None,
        plot=True,
        th=float(th) if th is not None else 1.0,
        subsample_rate_low=float(subsample_rate_low) if subsample_rate_low is not None else 0.2,
    )

    return data, cols, fig


@app.callback(
    Output("download_df", "data"),
    Input("btn_download", "n_clicks"),
    State("df_store", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, df_json):
    if not df_json:
        return no_update
    df = pd.read_json(df_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "scenario.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True)