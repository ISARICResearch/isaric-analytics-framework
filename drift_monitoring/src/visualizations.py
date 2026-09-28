from __future__ import annotations
from driftMonitoring import *
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def results_to_frame(results: List[Dict[str, Any]]) -> pd.DataFrame:
    # Convert a list of result dictionaries to a DataFrame.
    df = pd.DataFrame(results)
    if not df.empty and "end_date" in df.columns:
        df = df.sort_values("end_date").reset_index(drop=True)
    return df


def first_alert_summary(df: pd.DataFrame) -> pd.Series:
    # Return a compact summary of the first alert and final monitoring state.
    out = {
        "n_rows": len(df),
        "final_cusum": np.nan,
        "n_alerts": 0,
        "first_alert_date": pd.NaT,
        "first_alert_step": np.nan,
        "max_metric_value": np.nan,
    }
    if df.empty:
        return pd.Series(out)

    if "cusum" in df.columns:
        out["final_cusum"] = float(df["cusum"].iloc[-1])

    if "alert" in df.columns:
        alerts = df.loc[df["alert"] == 1]
        out["n_alerts"] = int(len(alerts))
        if not alerts.empty:
            out["first_alert_date"] = alerts["end_date"].iloc[0] if "end_date" in alerts.columns else pd.NaT
            out["first_alert_step"] = int(alerts["step"].iloc[0]) if "step" in alerts.columns else np.nan

    if "metric_value" in df.columns:
        out["max_metric_value"] = float(np.nanmax(df["metric_value"].to_numpy()))

    return pd.Series(out)


def continuous_result(results):
    x=[]
    median=[]
    q1=[]
    q3=[]
    cusum=[]
    cusum_date=[]
    for batch in results:
        x.append(batch['end_date'])
        median.append(batch['evidence'][0])
        q1.append(batch['evidence'][1])
        q3.append(batch['evidence'][2])
        cusum.append(batch['cusum'])
        cusum_date.append((batch['end_date'], batch['cusum']))


    x_original = x  
    #x=x_smooth
    #median=median_smooth
    #q1=q1_smooth
    #q3=q3_smooth
    return x,x_original, median, q1, q3, cusum, cusum_date

def binary_result(results):
    x=[]
    count=[]
    percentage=[]
    cusum=[]
    cusum_date=[]
    metric_value=[]
    for batch in results:
        metric_value.append(batch['metric_value'])
        x.append(batch['end_date'])
        count.append(batch['evidence'][0])
        percentage.append(batch['evidence'][1])
        cusum.append(batch['cusum'])
        cusum_date.append((batch['end_date'], batch['cusum']))
    return x, count, percentage, cusum, cusum_date

def continuous_figure(results, events,title, smooth=False,smooth_window=30,show_events=True):
    x, x_original, median, q1, q3, cusum, cusum_date = continuous_result(results)
    threshold = results[0]['threshold']

    ###########################
    #############################
    
    if smooth:
        median_plot = pd.Series(median).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()

        q1_plot = pd.Series(q1).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()

        q3_plot = pd.Series(q3).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
    else:
        median_plot = median
        q1_plot = q1
        q3_plot = q3


    ###########################
    ##############################


    # numeric x positions for plotting
    x_idx = list(range(len(x_original)))

    # dates only for labels
    dates = pd.to_datetime(x_original)

    # choose a small number of tick labels, evenly spaced
    n_ticks = min(8, len(x_idx))
    tick_idx = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
    tick_text = [dates[i].strftime("%b-%Y") for i in tick_idx]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4]
    )

    # Top plot: median + IQR
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=q3_plot,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=q1_plot,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,164,194,0.2)",
            name="IQR"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=median_plot,
            mode="lines",
            name="Median",
            line=dict(color="rgb(0,164,194)", width=4)
        ),
        row=1, col=1
    )

    # event markers: map event dates to nearest index
    if show_events:
        for date_str, label in events.items():
            date = pd.to_datetime(date_str)

            idx = min(
                range(len(dates)),
                key=lambda i: abs(dates[i] - date)
            )

            fig.add_trace(
                go.Scatter(
                    x=[x_idx[idx]],
                    y=[median[idx]],
                    mode="markers+text",
                    marker=dict(color="red", size=8),
                    text=[label],
                    textposition="top center",
                    showlegend=False
                ),
                row=1, col=1
            )

    # Bottom plot: CUSUM
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=cusum,
            mode="lines",
            name="Cusum",
            line=dict(color="rgb(0,164,194)", width=4)
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=threshold,
        line_color="red",
        line_dash="dot",
        line_width=2,
        row=2,
        col=1
    )

    # show dates only as selected tick labels
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_idx,
        ticktext=tick_text
    )

    fig.update_layout(
        title=title,
        xaxis_title="",
        xaxis2_title="Date",
        yaxis_title="Measure",
        yaxis2_title="CUSUM",
        legend_title_text="",
        showlegend=False
    )

    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "figure",
            "width": 1500,
            "height": 500,
            "scale": 1.5
        }
    }

    #fig.show(config=config)
    return fig

def binary_figure(results, events, title, smooth=False, smooth_window=30,show_events=True):
    
    x, count, percentage, cusum, cusum_date = binary_result(results)
    percentage = np.array(percentage)
    cusum = np.array(cusum)

    

    # Original dates for mapping / tick labels
    dates = pd.to_datetime(x)

    # Numeric x positions for the full timeline
    x_idx = np.arange(len(dates))

    # Top plot defaults to original data
    x_top_idx = x_idx
    dates_top = dates
    percentage_plot = percentage

    # Compress only the TOP plot
    if smooth:
        step = smooth_window

        x_top_idx_new = []
        dates_top_new = []
        percentage_new = []

        for start in range(0, len(percentage), step):
            end = min(start + step, len(percentage))
            mid = start + (end - start) // 2

            x_top_idx_new.append(x_idx[mid])
            dates_top_new.append(dates[mid])
            percentage_new.append(np.mean(percentage[start:end]))

        x_top_idx = np.array(x_top_idx_new)
        dates_top = np.array(dates_top_new)
        percentage_plot = np.array(percentage_new)

    complement = 100 - percentage_plot
    threshold = results[0]["threshold"]
    max_level=max(percentage_plot)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4]
    )

    # Top plot
    fig.add_trace(
        go.Bar(
            x=x_top_idx,
            y=percentage_plot,
            name="Observed",
            marker=dict(color="rgb(0,164,194)")
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=x_top_idx,
            y=complement,
            name="Complement",
            marker=dict(color="rgb(154,154,154)")
        ),
        row=1, col=1
    )

    fig.update_layout(barmode="stack")

    # Event markers on the TOP plot, mapped to the nearest TOP date
    # Event markers on the TOP plot
    #marker_levels = [15, 50, 85]

    marker_levels=[round(max_level*0.15),round(max_level*0.5),round(max_level*0.85)]

    if show_events:
        for j, (date_str, label) in enumerate(events.items()):
            date = pd.to_datetime(date_str)

            idx = min(
                range(len(dates_top)),
                key=lambda i: abs(dates_top[i] - date)
            )

            y_pos = marker_levels[j % len(marker_levels)]

            fig.add_trace(
                go.Scatter(
                    x=[x_top_idx[idx]],
                    y=[y_pos],
                    mode="markers+text",
                    marker=dict(color="red", size=8),
                    text=[label],
                    textposition="top center",
                    showlegend=False
                ),
                row=1, col=1
            )
    # Bottom plot: unchanged CUSUM
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=cusum,
            mode="lines",
            name="Cusum",
            line=dict(color="rgb(0,164,194)", width=4)
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=threshold,
        line_color="red",
        line_dash="dot",
        line_width=2,
        row=2,
        col=1
    )

    # Tick labels from the original timeline
    n_ticks = min(8, len(dates))
    tick_idx = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    tick_text = [dates[i].strftime("%b-%Y") for i in tick_idx]

    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_idx,
        ticktext=tick_text
    )

    fig.update_layout(
        title=title,
        xaxis_title="",
        xaxis2_title="Date",
        yaxis_title="Measure",
        yaxis2_title="CUSUM",
        legend_title_text="",
        showlegend=False
    )

    return fig
def fig_sampled_vs_full(
    data,
    value_col,
    kind,
    block_col="temp_block",
    sampled_col="sampled",
    title="Distribution by temporal block",
    y_axis_label=None,
):
    if y_axis_label is None:
        y_axis_label = "Percentage" if kind == "binary" else "Measure"

    fig = go.Figure()

    if kind == "continuous":
        fig.add_trace(
            go.Violin(
                x=data[block_col],
                y=data[value_col],
                name="Full cohort",
                box_visible=True,
                line_color="rgb(165,230,240)",
                points=False,
            )
        )

        sampled = data[data[sampled_col]]

        fig.add_trace(
            go.Violin(
                x=sampled[block_col],
                y=sampled[value_col],
                name="Sampled cohort",
                box_visible=True,
                line_color="rgb(0,164,194)",
                points=False,
            )
        )

        fig.update_layout(violinmode="group")
########        

    elif kind == "binary":
        blocks = []
        values_full = []
        values_sampled = []

        for db in data[block_col].dropna().unique():
            block_data = data[data[block_col] == db]

            if len(block_data) == 0:
                continue

            trues = (block_data[value_col] == True).sum()
            rounded_percentage = round(trues / len(block_data) * 100, 2)

            sampled_block = block_data.loc[block_data[sampled_col]]
            if len(sampled_block) > 0:
                sampled_trues = (sampled_block[value_col] == True).sum()
                rounded_sampled_percentage = round(sampled_trues / len(sampled_block) * 100, 2)
            else:
                rounded_sampled_percentage = 0

            blocks.append(db)
            values_full.append(rounded_percentage)
            values_sampled.append(rounded_sampled_percentage)

        x = np.arange(len(blocks))
        bar_width = 0.35

        # Full cohort, shifted left
        fig.add_trace(
            go.Bar(
                x=x - bar_width / 2,
                y=values_full,
                text=[f"{v:.1f}%" for v in values_full],
                textposition="inside",
                insidetextanchor="middle",
                name="Full cohort: True",
                offsetgroup="full",
                width=bar_width,
                #marker_color="rgb(120,120,120)",
                marker_color="rgb(165,230,240)",
            )
        )        
        false_full=100 - np.array(values_full)
        fig.add_trace(
            go.Bar(
                x=x - bar_width / 2,
                y=false_full,
                text=[f"{v:.1f}%" for v in false_full],
                textposition="inside",
                insidetextanchor="middle",                
                name="Full cohort: False",
                offsetgroup="full",
                width=bar_width,
                marker_color="rgb(190,190,190)",
                
            )
        )


        # Sampled cohort, shifted right
        fig.add_trace(
            go.Bar(
                x=x + bar_width / 2,
                y=values_sampled,
                text=[f"{v:.1f}%" for v in values_sampled],
                textposition="inside",
                insidetextanchor="middle",
                name="Sampled cohort: True",
                offsetgroup="sampled",
                width=bar_width,
                marker_color="rgb(0,164,194)",
            )
        )        
        false_sampled=100 - np.array(values_sampled)
        fig.add_trace(
            go.Bar(
                x=x + bar_width / 2,
                y=false_sampled,
                text=[f"{v:.1f}%" for v in false_sampled],
                textposition="inside",
                insidetextanchor="middle",
                name="Sampled cohort: False",
                offsetgroup="sampled",
                width=bar_width,
                marker_color="rgb(120,120,120)",
                #marker_color="rgb(165,230,240)",
            )
        )


        fig.update_layout(barmode="stack")
        fig.update_xaxes(tickmode="array", tickvals=x, ticktext=blocks)

    fig.update_layout(
        title=title,
        yaxis_title=y_axis_label,
        xaxis_title="Temporal block",
        template="plotly_white",
    )

    return fig

  