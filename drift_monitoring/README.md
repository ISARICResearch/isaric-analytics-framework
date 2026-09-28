# Drift Monitoring Framework

A small, self-contained example of reference-based Hellinger-CUSUM monitoring for temporal distributional drift.

## Overview

The repository contains one runnable Jupyter notebook that:

- generates synthetic patient-level surveillance data;
- compares consecutive patient batches with an initial reference distribution;
- accumulates Hellinger distances using a one-sided CUSUM;
- displays the simulated measurements and monitoring statistic inline; and
- allows the scenario parameters to be changed without requiring external data.

The example is based on a framework developed for adaptive clinical characterisation during infectious disease outbreaks. It is intended to demonstrate the method, not reproduce the restricted real-world clinical analysis.

## Repository structure

```text
.
├── simulation_example.ipynb   # Runnable synthetic-data example
├── src/
│   ├── driftMonitoring.py     # Simulation and monitoring functions
│   ├── visualizations.py      # Continuous and binary plots
│   └── ...                    # Supporting research modules
├── requirements.txt
├── LICENSE
└── README.md
```

Only `simulation_example.ipynb`, `src/driftMonitoring.py`, and `src/visualizations.py` are needed for the example.

## Installation

Create and activate a virtual environment, then install the requirements.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open `simulation_example.ipynb`, select the new `.venv` as the notebook kernel, and run all cells.

## Changing the simulation

Edit the `SCENARIO` dictionary in the notebook. The supplied example creates an abrupt change in a continuous measurement:

```python
SCENARIO = {
    "kind": "continuous",
    "scenario_type": "step_change",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "events": {"2024-07-01": "Step change"},
    "n_per_block_list": [1200, 1200],
    "values": [50, 70],
    "sds": [8, 8],
    "col_name": "measure",
}
```

The main fields are:

- `kind`: `continuous` or `binary`;
- `scenario_type`: `gradual_drift`, `step_change`, `transient_peak`, or `dispersion_increase`;
- `events`: dates separating the simulation periods;
- `n_per_block_list`: number of simulated patients in each period;
- `values`: continuous means or binary event probabilities; and
- `sds`: standard deviations for continuous variables.

For a binary scenario, set `kind` to `binary`, provide probabilities between 0 and 1 in `values`, and omit `sds`.

## Interpreting the figure

For continuous variables, the upper panel shows the batch median and interquartile range. For binary variables, it shows the observed event percentage. The lower panel shows the Hellinger-CUSUM statistic; the red dotted line is the alert threshold. Event labels indicate the changes defined in the scenario.

The figures are displayed in the notebook and are not exported to disk.

## Monitoring parameters

The example uses the parameters selected in the dissertation implementation:

```python
K = {"continuous": 0.35, "binary": 0.10}[SCENARIO["kind"]]
THRESHOLD = 10 * K
```

These values are provided as an illustrative research example and should be recalibrated for other data, batch sizes, or operational settings.

## Data availability

No patient-level data are included or required. All observations used by the example are generated synthetically when the notebook runs.

## Intended use

This repository is intended for research, education, and methodological demonstration. It is not a clinical decision-support system and should not be used to make individual patient-care decisions.

## License

Released under the MIT License.
