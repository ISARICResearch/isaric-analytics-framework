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

print('hello')



simulation_data = {}

# 1) Continuous gradual drift: 5 blocks
events_cont_gradual = {
    "2020-06-01": "Change 1",
    "2020-09-01": "Change 2",
    "2020-12-01": "Change 3",
   # "2021-03-01": "Change 4",
}

simulation_data["continuous_gradual_drift"] = generate_scenario(
    kind="continuous",
    scenario_type="gradual_drift",
    start_date="2020-03-16",
    end_date="2021-05-02",
    events=events_cont_gradual,
    n_per_block_list=[10000, 10000, 10000, 10000],
    values=[80, 60, 40, 20],
    col_name="measure"
)

# 2) Continuous step change: 2 blocks
events_cont_step = {
    "2020-10-01": "Step change",
}

simulation_data["continuous_step_change"] = generate_scenario(
    kind="continuous",
    scenario_type="step_change",
    start_date="2020-03-16",
    end_date="2021-05-02",
    events=events_cont_step,
    n_per_block_list=[30000, 20000],
    values=[75, 50],
    col_name="measure"
)

# 3) Continuous transient peak: 3 blocks
events_cont_peak = {
    "2020-10-01": "Peak begins",
    "2020-12-01": "Peak ends",
}

simulation_data["continuous_transient_peak"] = generate_scenario(
    kind="continuous",
    scenario_type="transient_peak",
    start_date="2020-03-16",
    end_date="2021-05-02",
    events=events_cont_peak,
    n_per_block_list=[15000, 10000, 25000],
    values=[40, 75, 40],
    col_name="measure"
)


events_high_noise = {
    "2020-10-01": "Variance increase 1",
    "2020-12-01": "Variance increase 2",
}

simulation_data["continuous_variance_change"] = generate_scenario(
    kind="continuous",
    scenario_type="dispersion_increase",
    start_date="2020-03-16",
    end_date="2021-05-02",
    events=events_high_noise,
    n_per_block_list=[15000, 10000, 25000],
    values=[50, 50, 50],
    sds=[10, 20, 30],
    col_name="measure"
)


events_no_drift = {
    "2020-10-01": "No change"
}

simulation_data["continuous_no_drift"] = generate_scenario(
    kind="continuous",
    scenario_type="gradual_drift",
    start_date="2020-03-16",
    end_date="2021-10-02",
    events=events_no_drift,
    n_per_block_list=[20000,40000],
    values=[50,50],
    sds=[30,30],
    col_name="measure"
)

'''
events_no_drift = {
    "2020-10-01": "No change"
}

simulation_data["continuous_noise"] = generate_scenario(
    kind="continuous",
    scenario_type="noise",
    start_date="2020-03-16",
    end_date="2021-05-02",
    events=events_no_drift,
    n_per_block_list=[20000,20000],
    values=[95,95],
    sds=[30,30],
    col_name="measure"
)

events_cont_gradual_1change = {
    "2020-06-01": "Change 1",

}

simulation_data["continuous_gradual_drift_1"] = generate_scenario(
    kind="continuous",
    scenario_type="gradual_drift",
    start_date="2020-01-01",
    end_date="2020-12-31",
    events=events_cont_gradual_1change,
    n_per_block_list=[10000, 10000],
    values=[60, 40],
    col_name="measure"
)
simulation_data["continuous_gradual_drift_2"] = generate_scenario(
    kind="continuous",
    scenario_type="gradual_drift",
    start_date="2020-01-01",
    end_date="2020-12-31",
    events=events_cont_gradual_1change,
    n_per_block_list=[10000, 10000],
    values=[60, 50],
    col_name="measure"
)

'''
# 1) Binary gradual drift: 5 blocks
events_bin_gradual = {
    "2020-12-01": "Phase 1",
    
}

simulation_data["binary_gradual_drift"] = generate_scenario(
    kind="binary",
    scenario_type="gradual_drift",
    start_date="2020-01-01",
    end_date="2021-12-31",
    events=events_bin_gradual,
    n_per_block_list=[8000, 8000],
    values=[0.10, 0.30],
    col_name="symptom"
)


# Binary step_change
events_binary_step_change = {
    "2020-12-01": "Phase 1",   
}
simulation_data["binary_step_change"] = generate_scenario(
    kind="binary",
    scenario_type="step_change",
    start_date="2020-01-01",
    end_date="2021-12-31",
    events=events_binary_step_change,
    n_per_block_list=[8000, 8000],
    values=[0.60, 0.35],
    col_name="symptom"
)



# 3) binary transient peak: 3 blocks
events_bin_peak = {
    "2020-12-01": "Peak begins",
    "2021-01-31": "Peak ends",
}

simulation_data["binary_transient_peak"] = generate_scenario(
    kind="binary",
    scenario_type="transient_peak",
    start_date="2020-01-01",
    end_date="2021-12-31",
    events=events_bin_peak,
    n_per_block_list=[6000, 6000, 6000],
    values=[0.2, 0.4, 0.15],
    col_name="symptom"
)


events_no_drift = {
    "2020-10-01": "No change"
}

simulation_data["binary_no_drift"] = generate_scenario(
    kind="binary",
    scenario_type="gradual_drift",
    start_date="2020-03-16",
    end_date="2021-10-02",
    events=events_no_drift,
    n_per_block_list=[8000,8000],
    values=[0.50,0.50],
    col_name="symptom"
)

'''
events_no_drift = {
    "2020-10-01": "No change"
}

simulation_data["binary_noise"] = generate_scenario(
    kind="binary",
    scenario_type="noise",
    start_date="2020-03-16",
    end_date="2021-10-02",
    events=events_no_drift,
    n_per_block_list=[20000,20000],
    values=[0.50,0.50],
    col_name="symptom"
)
'''




# Global monitoring parameters used in the dissertation examples
k_values={'continuous': 0.35,"binary":0.1}
th_values={'continuous': 10*k_values['continuous'],"binary":10*k_values['binary']}
simulation_results = {}

simulation_meta = {
    "continuous_gradual_drift": {
        "title": "Continuous Variable: Gradual Upward Drift",
        "kind": "continuous",
        "value_col": "measure",
        "events": events_cont_gradual
    },
    "continuous_step_change": {
        "title": "Continuous Variable: Abrupt Step Change",
        "kind": "continuous",
        "value_col": "measure",
        "events": events_cont_step
    },
    "continuous_transient_peak": {
        "title": "Continuous Variable: Transient Peak",
        "kind": "continuous",
        "value_col": "measure",
        "events": events_cont_peak
    },
    "continuous_variance_change": {
        "title": "Continuous Variable: Increased Variability",
        "kind": "continuous",
        "value_col": "measure",
        "events": events_high_noise
    },
    "continuous_no_drift": {
        "title": "Continuous Variable: Stable Baseline (No Drift)",
        "kind": "continuous",
        "value_col": "measure",
        "events": events_no_drift
    },

    "binary_gradual_drift": {
        "title": "Binary Variable: Gradual Increase in Event Rate",
        "kind": "binary",
        "value_col": "symptom",
        "events": events_bin_gradual
    },
    "binary_step_change": {
        "title": "Binary Variable: Abrupt Increase in Event Rate",
        "kind": "binary",
        "value_col": "symptom",
        "events": events_binary_step_change
    },
    "binary_transient_peak": {
        "title": "Binary Variable: Transient Increase in Event Rate",
        "kind": "binary",
        "value_col": "symptom",
        "events": events_bin_peak
    },
    "binary_no_drift": {
        "title": "Binary Variable: Stable Event Rate (No Drift)",
        "kind": "binary",
        "value_col": "symptom",
        "events": events_no_drift
    },
}

output_dir="../outputs/"

for name, sim_data in simulation_data.items():
    sim_data['subjid'] = range(1, len(sim_data) + 1)
    
    kind = simulation_meta[name]["kind"]
    value_col = simulation_meta[name]["value_col"]
    ev = simulation_meta[name]["events"]
    k=k_values[kind]
    th=th_values[kind]
    title=simulation_meta[name]['title']

    results = rolling_metric_fixed_baseline_cusum(
        data=sim_data,
        value_col=value_col,
        date_col="date_admit",
        events=ev,
        batch=100,
        baseline_batches=3,
        n_bins=10,
        k=k,
        metric_name=None,
        th=th,
        subsample_rate_low=0.2,
        stable_patience=5
    )
    
    df_results = results_to_frame(results)
    simulation_results[name] = df_results

    print(f"\n{name}")
    
    if simulation_meta[name]["kind"]=='continuous':
        fig=continuous_figure(results, events=ev,title=title)
        fig.show()
        
       
    else:
        fig=binary_figure(results,ev,title=title)

    #fig.show()
