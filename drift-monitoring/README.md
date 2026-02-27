# Drift Monitoring Framework  
Batch-Based Distributional Drift Detection with Adaptive Sampling

## Overview

This module implements a reusable analytical pipeline for detecting pattern shifts in patient-level data streams during infectious disease outbreaks. The method combines distributional drift metrics with cumulative sum (CUSUM) methodology to detect both gradual and abrupt changes in clinical and epidemiological patterns.

The framework is designed to:

- Detect sustained distributional change relative to a reference baseline  
- Distinguish meaningful shifts from random fluctuation  
- Adapt sampling intensity based on system stability  
- Rebuild baselines after detected change  
- Remain modular, reusable, and disease-agnostic  

This method aligns with ISARIC’s philosophy of reusable analytical pipelines (RAPs) and scalable outbreak analytics.

---

## Methodological Components

### 1. Baseline-Referenced Batch Monitoring

Incoming patient data are processed in sequential batches (e.g., every 100 patients). Each batch is compared to an initial reference distribution constructed from early data.

Drift is quantified using:

- **Hellinger distance** for binary variables  
- **Jensen–Shannon divergence** for continuous variables  

These metrics measure distributional change rather than relying solely on summary statistics such as means.

---

### 2. Cumulative Drift Detection (CUSUM)

To detect gradual changes, the framework applies a one-sided cumulative sum procedure:

- A calibrated noise parameter (`k`) filters minor fluctuations.
- Drift above this noise floor accumulates over time.
- An alert is triggered when cumulative drift exceeds a predefined threshold (`th`).

This approach enables detection of sustained small shifts that would not be visible through batch-to-batch comparison alone.

---

### 3. Adaptive Sampling

To reduce operational burden during stable periods:

- Sampling intensity is automatically reduced when sustained stability is detected.
- Full data capture is restored immediately when an alert is triggered.

This allows resource-aware surveillance without compromising responsiveness.

---

### 4. Re-Baselining

Following an alert, the reference distribution is rebuilt from a configurable number of subsequent batches (`baseline_batches`), enabling adaptation to evolving disease characterisation and changing outbreak dynamics.

---

## Key Parameters

- `batch`: Number of patients per monitoring batch  
- `baseline_batches`: Number of batches used to construct or rebuild the reference  
- `k`: Noise floor parameter controlling sensitivity  
- `th`: CUSUM alert threshold  
- `stable_patience`: Number of consecutive stable batches required before subsampling  
- `subsample_rate_low`: Reduced sampling fraction during stable periods  

Parameter calibration can be performed using bootstrap or simulation-based approaches.

---

## Validation

The framework has been evaluated through:

- Simulation experiments assessing abrupt and gradual drift detection  
- Application to COVID-19 clinical datasets within the ISARIC platform  

Results demonstrate reliable detection of sustained change while dynamically modulating data capture intensity.

---

## Intended Use

This module supports:

- Monitoring changing disease patterns over time  
- Identifying emerging clinical phenotypes  
- Detecting shifts in severity or symptom profiles  
- Supporting adaptive evidence generation in outbreak settings  

The method is disease-agnostic and can be validated across multiple ISARIC datasets.

---

## Status

Active development. Designed for modular integration within the ISARIC Reusable Analytics Framework.

---

## License

Released under the MIT License as part of the ISARIC Reusable Analytics Framework.
