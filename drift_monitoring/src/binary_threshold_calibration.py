"""Calibrate Hellinger-CUSUM allowance and threshold parameters.

The binary functions reproduce the real-world symptom-monitoring design:
100-admission batches, 10 initial reference batches, 2,527 post-baseline
batches, 20% sampling after five stable batches, and re-baselining after an
alert.  The continuous functions use the same design and the pooled-quantile
binning currently used by ``driftMonitoring.empirical_distributions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_hellinger(reference_events, reference_n, batch_events, batch_n):
    """Hellinger distance between Bernoulli distributions represented by counts."""
    p_reference = reference_events / reference_n
    p_batch = batch_events / batch_n
    return float(np.sqrt(
        ((np.sqrt(p_reference) - np.sqrt(p_batch)) ** 2
         + (np.sqrt(1 - p_reference) - np.sqrt(1 - p_batch)) ** 2) / 2
    ))


def probability_schedule(
    scenario: str,
    baseline_prevalence: float,
    changed_prevalence: float,
    n_monitoring_batches: int,
) -> tuple[np.ndarray, int | None]:
    """Return event probabilities and the first batch with a true change."""
    change_batch = n_monitoring_batches // 2
    probabilities = np.full(n_monitoring_batches, baseline_prevalence)

    if scenario == "no_drift":
        return probabilities, None

    if scenario == "step_change":
        probabilities[change_batch:] = changed_prevalence
        return probabilities, change_batch + 1

    if scenario == "gradual_change":
        gradual_batches = n_monitoring_batches // 4
        end_batch = min(change_batch + gradual_batches, n_monitoring_batches)
        probabilities[change_batch:end_batch] = np.linspace(
            baseline_prevalence,
            changed_prevalence,
            end_batch - change_batch,
        )
        probabilities[end_batch:] = changed_prevalence
        return probabilities, change_batch + 1

    raise ValueError("scenario must be 'no_drift', 'step_change', or 'gradual_change'.")


def simulate_run(
    scenario: str,
    baseline_prevalence: float,
    changed_prevalence: float,
    threshold: float,
    *,
    k: float = 0.10,
    batch_size: int = 100,
    baseline_batches: int = 10,
    n_monitoring_batches: int = 2527,
    stable_patience: int = 5,
    reduced_sampling_rate: float = 0.20,
    seed: int = 1,
) -> dict:
    """Run one binary calibration simulation using the monitoring rules."""
    rng = np.random.default_rng(seed)
    probabilities, true_change_step = probability_schedule(
        scenario,
        baseline_prevalence,
        changed_prevalence,
        n_monitoring_batches,
    )

    # Underlying full batches are fixed for every threshold in a replicate.
    baseline_counts = rng.binomial(batch_size, baseline_prevalence, baseline_batches)
    full_batch_counts = rng.binomial(batch_size, probabilities)

    reference_events = int(baseline_counts.sum())
    reference_n = baseline_batches * batch_size
    cusum = 0.0
    sampling_rate = 1.0
    rebaseline_counts = []
    rebaselining = False
    stable_history = []
    alert_steps = []

    for step, full_events in enumerate(full_batch_counts, start=1):
        if rebaselining:
            rebaseline_counts.append(int(full_events))
            if len(rebaseline_counts) == baseline_batches:
                reference_events = int(sum(rebaseline_counts))
                reference_n = baseline_batches * batch_size
                cusum = 0.0
                rebaseline_counts = []
                rebaselining = False
                sampling_rate = 1.0
            continue

        current_n = batch_size
        current_events = int(full_events)

        if sampling_rate < 1.0:
            current_n = int(batch_size * sampling_rate)
            current_events = int(rng.hypergeometric(
                ngood=int(full_events),
                nbad=batch_size - int(full_events),
                nsample=current_n,
            ))

        distance_value = binary_hellinger(
            reference_events,
            reference_n,
            current_events,
            current_n,
        )
        cusum = max(0.0, cusum + distance_value - k)
        in_alert = cusum >= threshold

        if in_alert:
            alert_steps.append(step)
            rebaselining = True
            sampling_rate = 1.0
            stable_history = []
        else:
            stable_history.append(False)
            stable_history = stable_history[-stable_patience:]
            if len(stable_history) == stable_patience:
                sampling_rate = reduced_sampling_rate

    alerts_after_change = [
        step for step in alert_steps
        if true_change_step is not None and step >= true_change_step
    ]
    first_detection_step = min(alerts_after_change) if alerts_after_change else np.nan

    return {
        "scenario": scenario,
        "baseline_prevalence": baseline_prevalence,
        "changed_prevalence": changed_prevalence,
        "threshold": threshold,
        "k": k,
        "n_monitoring_batches": n_monitoring_batches,
        "number_of_alerts": len(alert_steps),
        "any_alert": bool(alert_steps),
        "detected_after_change": bool(alerts_after_change),
        "detection_delay_batches": (
            first_detection_step - true_change_step
            if pd.notna(first_detection_step)
            else np.nan
        ),
    }


def calibrate_thresholds(
    thresholds=(0.50, 0.75, 1.00, 1.25, 1.50, 2.00),
    baseline_prevalences=(0.05, 0.10, 0.30, 0.50),
    n_replicates: int = 100,
    n_monitoring_batches: int = 2527,
    binary_k: float = 0.10,
    k_values: tuple[float, ...] | None = (0.10, 0.25, 0.5),
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jointly evaluate binary CUSUM allowance values and thresholds.

    ``binary_k`` preserves the earlier single-value interface. Pass
    ``k_values``, for example ``(0.05, 0.10, 0.15)``, to compare allowances.
    """
    rng = np.random.default_rng(seed)
    run_results = []
    if k_values is None:
        k_values = (binary_k,)

    for prevalence in baseline_prevalences:
        changed_prevalence = min(prevalence + 0.20, 0.95)
        for scenario in ("no_drift", "step_change", "gradual_change"):
            for _ in range(n_replicates):
                replicate_seed = int(rng.integers(1, np.iinfo(np.int32).max))
                for k in k_values:
                    for threshold in thresholds:
                        run_results.append(simulate_run(
                            scenario=scenario,
                            baseline_prevalence=prevalence,
                            changed_prevalence=changed_prevalence,
                            threshold=threshold,
                            k=k,
                            n_monitoring_batches=n_monitoring_batches,
                            seed=replicate_seed,
                        ))

    run_results = pd.DataFrame(run_results)
    no_drift = run_results.query("scenario == 'no_drift'")
    changed = run_results.query("scenario != 'no_drift'")

    summary = (
        no_drift.groupby(["k", "threshold"], as_index=False)
        .agg(
            **{
                "False-alert runs (%)": ("any_alert", lambda x: 100 * x.mean()),
                "Mean false alerts": ("number_of_alerts", "mean"),
            }
        )
        .merge(
            changed.groupby(["k", "threshold"], as_index=False).agg(
                **{
                    "Detection rate (%)": ("detected_after_change", lambda x: 100 * x.mean()),
                    "Median detection delay (batches)": ("detection_delay_batches", "median"),
                }
            ),
            on=["k", "threshold"],
        )
    )

    return run_results, summary


def continuous_hellinger(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int,
) -> float:
    """Hellinger distance using pooled quantile bins.

    This intentionally matches the current continuous implementation in
    ``driftMonitoring.empirical_distributions``: bin edges are recalculated
    from the pooled reference and current observations for each comparison.
    """
    pooled = np.concatenate([reference, current])
    bins = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    if bins.size < 2:
        return 0.0

    reference_probabilities = np.histogram(reference, bins=bins)[0].astype(float)
    current_probabilities = np.histogram(current, bins=bins)[0].astype(float)
    reference_probabilities /= reference_probabilities.sum()
    current_probabilities /= current_probabilities.sum()

    return float(np.sqrt(
        np.sum((np.sqrt(reference_probabilities) - np.sqrt(current_probabilities)) ** 2)
        / 2
    ))


def continuous_mean_schedule(
    scenario: str,
    baseline_mean: float,
    changed_mean: float,
    n_monitoring_batches: int,
) -> tuple[np.ndarray, int | None]:
    """Return batch means and the first batch with a true distributional change."""
    change_batch = n_monitoring_batches // 2
    means = np.full(n_monitoring_batches, baseline_mean)

    if scenario == "no_drift":
        return means, None
    if scenario == "step_change":
        means[change_batch:] = changed_mean
        return means, change_batch + 1
    if scenario == "gradual_change":
        gradual_batches = n_monitoring_batches // 4
        end_batch = min(change_batch + gradual_batches, n_monitoring_batches)
        means[change_batch:end_batch] = np.linspace(
            baseline_mean,
            changed_mean,
            end_batch - change_batch,
        )
        means[end_batch:] = changed_mean
        return means, change_batch + 1
    raise ValueError("scenario must be 'no_drift', 'step_change', or 'gradual_change'.")


def simulate_continuous_run(
    scenario: str,
    threshold: float,
    *,
    k: float,
    n_bins: int = 10,
    baseline_mean: float = 0.0,
    changed_mean: float = 0.5,
    standard_deviation: float = 1.0,
    batch_size: int = 100,
    baseline_batches: int = 10,
    n_monitoring_batches: int = 2527,
    stable_patience: int = 5,
    reduced_sampling_rate: float = 0.20,
    seed: int = 1,
) -> dict:
    """Run one continuous calibration simulation with the monitoring rules.

    The default change is a 0.5-standard-deviation mean shift. Change
    ``changed_mean`` and ``standard_deviation`` to test an alternative effect
    size, while retaining the same batch and binning design.
    """
    rng = np.random.default_rng(seed)
    means, true_change_step = continuous_mean_schedule(
        scenario,
        baseline_mean,
        changed_mean,
        n_monitoring_batches,
    )
    reference = rng.normal(
        baseline_mean,
        standard_deviation,
        size=baseline_batches * batch_size,
    )
    full_batches = rng.normal(
        loc=means[:, None],
        scale=standard_deviation,
        size=(n_monitoring_batches, batch_size),
    )

    cusum = 0.0
    sampling_rate = 1.0
    rebaseline_batches: list[np.ndarray] = []
    rebaselining = False
    stable_batches = 0
    alert_steps: list[int] = []
    distance_values: list[float] = []

    for step, full_batch in enumerate(full_batches, start=1):
        if rebaselining:
            rebaseline_batches.append(full_batch)
            if len(rebaseline_batches) == baseline_batches:
                reference = np.concatenate(rebaseline_batches)
                cusum = 0.0
                rebaseline_batches = []
                rebaselining = False
                sampling_rate = 1.0
            continue

        current = full_batch
        if sampling_rate < 1.0:
            sampled_size = int(batch_size * sampling_rate)
            current = rng.choice(full_batch, size=sampled_size, replace=False)

        distance_value = continuous_hellinger(reference, current, n_bins=n_bins)
        distance_values.append(distance_value)
        cusum = max(0.0, cusum + distance_value - k)

        if cusum >= threshold:
            alert_steps.append(step)
            rebaselining = True
            sampling_rate = 1.0
            stable_batches = 0
        else:
            stable_batches += 1
            if stable_batches >= stable_patience:
                sampling_rate = reduced_sampling_rate

    alerts_after_change = [
        step for step in alert_steps
        if true_change_step is not None and step >= true_change_step
    ]
    first_detection_step = min(alerts_after_change) if alerts_after_change else np.nan

    return {
        "scenario": scenario,
        "n_bins": n_bins,
        "k": k,
        "threshold": threshold,
        "mean_batch_hellinger": float(np.mean(distance_values)),
        "number_of_alerts": len(alert_steps),
        "any_alert": bool(alert_steps),
        "detected_after_change": bool(alerts_after_change),
        "detection_delay_batches": (
            first_detection_step - true_change_step
            if pd.notna(first_detection_step)
            else np.nan
        ),
    }


def calibrate_continuous_parameters(
    k_values=(0.20, 0.25, 0.30, 0.35, 0.40),
    thresholds=(1.50, 2.00, 2.50, 3.00, 3.50, 4.00),
    n_bins_values=(5, 10, 15),
    n_replicates: int = 25,
    n_monitoring_batches: int = 2527,
    baseline_mean: float = 0.0,
    changed_mean: float = 0.5,
    standard_deviation: float = 1.0,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jointly calibrate continuous-variable bin count, allowance, and threshold.

    Each bin count is evaluated under no drift, a step change, and a gradual
    mean shift. The returned summary supports selecting a design with an
    acceptable false-alert frequency and detection delay.
    """
    rng = np.random.default_rng(seed)
    run_results = []

    for n_bins in n_bins_values:
        for scenario in ("no_drift", "step_change", "gradual_change"):
            for _ in range(n_replicates):
                replicate_seed = int(rng.integers(1, np.iinfo(np.int32).max))
                for k in k_values:
                    for threshold in thresholds:
                        run_results.append(simulate_continuous_run(
                            scenario=scenario,
                            threshold=threshold,
                            k=k,
                            n_bins=n_bins,
                            baseline_mean=baseline_mean,
                            changed_mean=changed_mean,
                            standard_deviation=standard_deviation,
                            n_monitoring_batches=n_monitoring_batches,
                            seed=replicate_seed,
                        ))

    run_results = pd.DataFrame(run_results)
    no_drift = run_results.query("scenario == 'no_drift'")
    changed = run_results.query("scenario != 'no_drift'")
    grouping = ["n_bins", "k", "threshold"]

    summary = (
        no_drift.groupby(grouping, as_index=False)
        .agg(
            **{
                "Mean no-drift Hellinger": ("mean_batch_hellinger", "mean"),
                "False-alert runs (%)": ("any_alert", lambda x: 100 * x.mean()),
                "Mean false alerts": ("number_of_alerts", "mean"),
            }
        )
        .merge(
            changed.groupby(grouping, as_index=False).agg(
                **{
                    "Detection rate (%)": ("detected_after_change", lambda x: 100 * x.mean()),
                    "Median detection delay (batches)": ("detection_delay_batches", "median"),
                }
            ),
            on=grouping,
        )
    )

    return run_results, summary


if __name__ == "__main__":
    '''
    runs, summary = calibrate_thresholds(n_replicates=5)
    print(summary.to_string(index=False))
    runs.to_csv("binary_threshold_calibration_runs.csv", index=False)
    summary.to_csv("binary_threshold_calibration_summary.csv", index=False)
    '''
    '''
    continuous_runs, continuous_summary = calibrate_continuous_parameters(
    n_bins_values=(10,),
    k_values=(0.1, 0.2, 0.35, 0.5),
    thresholds=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    n_replicates=5,

    n_monitoring_batches=2527)
    continuous_summary.sort_values(["n_bins", "False-alert runs (%)", "Median detection delay (batches)"])
    print(continuous_summary.to_string(index=False))
    continuous_summary.to_csv("continuous_threshold_calibration_summary.csv", index=False)'''
    '''
    continuous_runs, continuous_summary = calibrate_continuous_parameters(
    n_bins_values=(2,5,10),
    k_values=(0.10,0.35,0.5),
    thresholds=(1.0,3.5,5.0),
    n_replicates=10,
    n_monitoring_batches=2500)
    continuous_summary.sort_values(["n_bins", "False-alert runs (%)", "Median detection delay (batches)"])
    print(continuous_summary.to_string(index=False))'''


    runs, summary = calibrate_thresholds(n_replicates=10, k_values=(0.10,0.15,0.2),
    thresholds=(1.0,1.5,2.0),)
    print(summary.to_string(index=False))
    runs.to_csv("binary_threshold_calibration_runs.csv", index=False)
    summary.to_csv("binary_threshold_calibration_summary.csv", index=False)
