from __future__ import annotations
from driftMonitoring import *
from visualizations import *
import random
import warnings
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import fisher_exact, ks_2samp
from statsmodels.stats.proportion import proportion_confint
from datetime import datetime
import isaricanalytics.IsaricAnalytics as ia
import isaricanalytics.IsaricDraw as idw


def get_results(data,events,kind,k,th,varibale_x,label_x):
    data_x=data[['subjid','date_admit',varibale_x]].dropna()
    #k=0.1

    results = rolling_metric_fixed_baseline_cusum(
        data=data_x, 
        value_col=varibale_x,
        date_col="date_admit",
        events=events,
        batch=100,
        baseline_batches=10,
        n_bins=10,
        k=k,              # example for a distance metric; tune from stable period
        metric_name=None,    # uses first metric returned by your metrics function
        th=th,
        subsample_rate_low=0.2
    )
    data_blocks = make_temporal_blocks(
        data_x,
        results,
        date_col="date_admit",
        subjid_col="subjid"
    )    
    comparison = get_comparison_table(data_blocks,kind=kind,var_name=varibale_x,var_label=label_x)
    if kind=="binary":
        fig=binary_figure(results,events,title=varibale_x,smooth=True,smooth_window=20)
        #fig.show()
    elif kind=="continuous":
        fig=continuous_figure(results,events,title=varibale_x,smooth=True,smooth_window=20)
        #fig.show()

    fig_s_vs_f = fig_sampled_vs_full(
        data_blocks,
        kind=kind,
        value_col=varibale_x,
        title=f"{label_x} distribution by temporal block"
    )

    return results, data_blocks, comparison, fig, fig_s_vs_f

if __name__ == "__main__":
    # Example only. The original dissertation data are not included in this
    # public repository. Replace these paths with your own local/private data.
    data_dir = Path("path/to/private/data")
    output_dir = Path("../outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = "your_patient_level_dataset.csv"
    data = pd.read_csv(data_dir / input_file)

    # Expected columns include:
    # subjid, date_admit, country_iso, and the variables passed to get_results.
    data = data.loc[data["country_iso"] == "GBR"]
    data["date_admit"] = pd.to_datetime(data["date_admit"], errors="coerce")
    data = data.dropna(subset=["date_admit"])
    data = data.sort_values("date_admit", ascending=True).reset_index(drop=True)

    events = {
        "2020-03-16": "closures",
        "2020-06-08": "Quarantine for arrivals",
        "2020-12-08": "First Pfizer vaccine dose (UK)",
        "2021-02-15": "15M people vaccinated",
        "2021-06-01": "Delta becomes dominant in UK",
        "2021-09-20": "Booster programme begins",
    }

    variables = [
        ["continuous", 0.35, 10 * 0.35, "demog_age", "Age"],
        ["binary", 0.10, 10 * 0.10, "comps_ards", "ARDS"],
    ]

    for kind, k, th, variable_x, label_x in variables:
        results, data_blocks, comparison, fig, fig_s_vs_f = get_results(
            data=data,
            events=events,
            kind=kind,
            k=k,
            th=th,
            varibale_x=variable_x,
            label_x=label_x,
        )
        comparison.to_csv(
            output_dir / f"{label_x.replace(' ', '_')}_comparison.csv",
            index=False,
        )
