import warnings
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ks_2samp
import plotly.express as px
from scipy.spatial import distance
from statsmodels.stats.proportion import proportion_confint
from scipy import stats
from scipy.stats import fisher_exact
import numpy as np
import pandas as pd
from scipy.spatial import distance
from plotly.subplots import make_subplots
from functools import reduce
warnings.filterwarnings("ignore")



############################
# --- metric functions ---
############################
#binary
def HellingerDistance(a, b, dropna=True):
    """
    Hellinger distance between two binary (0/1) samples, ignoring NaNs.

    Parameters
    ----------
    a, b : array-like of {0, 1, np.nan}
        Two samples of a Bernoulli variable.
    dropna : bool, default True
        If True, drop NaNs before computing.

    Returns
    -------
    h : float in [0, 1]
        Hellinger distance between Bernoulli(p1) and Bernoulli(p2).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")

    # Validate binary (allowing float 0.0/1.0)
    if not np.isin(a, [0.0, 1.0]).all() or not np.isin(b, [0.0, 1.0]).all():
        raise ValueError("Inputs must be binary (0/1/NaN).")

    p1 = a.mean()  # proportion of 1s
    p2 = b.mean()

    # Hellinger distance for Bernoulli(p1) vs Bernoulli(p2)
    # H = (1/sqrt(2)) * sqrt( (sqrt(p1)-sqrt(p2))^2 + (sqrt(1-p1)-sqrt(1-p2))^2 )
    h = (1/np.sqrt(2)) * np.sqrt((np.sqrt(p1) - np.sqrt(p2))**2 +
                                 (np.sqrt(1 - p1) - np.sqrt(1 - p2))**2)
    return float(h)

def CohenH(a, b, dropna=True):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")

    p1 = a.mean()  # proportion of 1s
    p2 = b.mean()

    # Cohen's h
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    h=np.abs(h)
    return float(h)

def prevalenceChange(a, b, dropna=True):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")

    p1 = a.mean()  # proportion of 1s
    p2 = b.mean()

    change = abs(p1 - p2)
    return float(change)

def WilsonCI(a, dropna=True, alpha=0.05):

    a = np.asarray(a, dtype=float)

    if dropna:
        a = a[~np.isnan(a)]

    if a.size == 0:
        raise ValueError("Input must contain at least one non-NaN value.")

    p = a.mean()  # proportion of 1s
    n = a.size

    z = stats.norm.ppf(1 - alpha / 2)
    denominator = 1 + (z**2) / n
    center = p + (z**2) / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)

    lower_bound = (center - margin) / denominator
    upper_bound = (center + margin) / denominator

    return float(upper_bound)-float(lower_bound)

def FisherExactTest(a, b, dropna=True):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")

    # Construct contingency table
    table = np.array([
        [np.sum(a == 1), np.sum(a == 0)],
        [np.sum(b == 1), np.sum(b == 0)]
    ])

    _, p_value = stats.fisher_exact(table)
    return float(p_value)
#continuos
def JSDivergence(a, b, n_bins=10, dropna=True):
    """
    Jensen–Shannon distance between two 1-D arrays using shared quantile bins.

    Parameters
    ----------
    a, b : array-like
        Raw numeric data (e.g., ages, heart rates).
    n_bins : int, default 10
        Number of quantile bins (deciles=10, quartiles=4, etc.). 
        Actual bins may be fewer if values are tied.
    dropna : bool, default True
        Drop NaNs before computing.

    Returns
    -------
    jsd : float
        Jensen–Shannon distance.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

    # Guard rails
    if a.size == 0 or b.size == 0:
        raise ValueError("Inputs must contain at least one non-NaN value each.")

    pooled = np.concatenate([a, b])

    # Quantile breakpoints (shared across a and b)
    qs = np.linspace(0, 1, n_bins + 1)
    cutoffs = np.quantile(pooled, qs)
    # Ensure strictly increasing edges
    cutoffs = np.unique(cutoffs)

    # If all values identical → only one unique edge left
    if cutoffs.size < 2:
        # Distributions are identical (all mass in one bin)
        return 0.0

    # Bin both arrays using the same edges; drop duplicated edges if any remain
    bins_a = pd.cut(a, bins=cutoffs, include_lowest=True, duplicates="drop")
    bins_b = pd.cut(b, bins=cutoffs, include_lowest=True, duplicates="drop")

    # Establish the full category index (all bins)
    all_bins = pd.Categorical(pd.cut(pooled, bins=cutoffs, include_lowest=True, duplicates="drop")).categories

    # Counts including zero-count bins
    count_a = pd.value_counts(bins_a, sort=False).reindex(all_bins, fill_value=0)
    count_b = pd.value_counts(bins_b, sort=False).reindex(all_bins, fill_value=0)

    # Convert to probabilities
    p = (count_a / count_a.sum()).to_numpy(dtype=float)
    q = (count_b / count_b.sum()).to_numpy(dtype=float)

    # JSD (SciPy normalizes if not exactly summing to 1 due to float)
    jsd = float(distance.jensenshannon(p, q))
    return jsd

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

    shift = abs(med_a - med_b)/med_b
    return float(shift)

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

    shift = abs(iqr_a - iqr_b)/iqr_b
    return float(shift)

def KStest(a, b, dropna=True):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if dropna:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

    if a.size == 0 or b.size == 0:
        raise ValueError("Both inputs must contain at least one non-NaN value.")
    
    ks_stat=stats.kstest(a, b)
    
    return ks_stat.statistic

############################
# --- metric grouping by variable type functions ---
############################

def binary_metrics(a, b):
    hd = HellingerDistance(a, b)
    cohen=CohenH(a, b)
    preval=prevalenceChange(a, b)
    #wilsonCI=WilsonCI(a)
    Fisher=FisherExactTest(a, b)
    return {
        "hellinger_distance": hd,
        "cohen_h": cohen,
        "prevalence_change": preval,
        "fisher_exact": Fisher,
    }

def continuous_metrics(a, b, n_bins=10):
    jsd = JSDivergence(a, b, n_bins=n_bins)
    med_shift=MedianShift(a,b)
    iqr_shift=IQRShift(a,b)
    ks=KStest(a,b)
    return {
        "js_divergence": jsd,
        "median_shift": med_shift,
        "iqr_shift": iqr_shift,
        "ks_test": ks,
    }

    
############################
# --- plotting functions ---
############################
def basic_result_plots(results,metric):
    res_df = pd.DataFrame(results)
    # use the midpoint of each batch as the x axis
    if "start_date" in res_df.columns and "end_date" in res_df.columns:
        res_df["mid_date"] = res_df[["start_date","end_date"]].mean(axis=1)
    else:
        res_df["mid_date"] = res_df[["week_start","week_end"]].mean(axis=1)

    fig = px.line(
        res_df,
        x="mid_date",
        y=metric,
        title=metric,
        range_y=[0, 1]
    )
    return fig  


def fig_percentage_bar(arr):
    
    arr = np.asarray(arr, dtype=float)

    if arr.shape[1] != 2:
        raise ValueError("arr must be of shape (n, 2): [count, percentage]")

    counts = arr[:, 0]
    perc = arr[:, 1]
    complement = 100 - perc

    x = np.arange(1, len(arr) + 1)  # 1,2,3,...

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x,
            y=perc,
            name="Percentage",
            customdata=np.column_stack((counts, perc)),
            hovertemplate=(
                "Count: %{customdata[0]}<br>"
                "Percentage: %{customdata[1]:.1f}%<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Bar(
            x=x,
            y=complement,
            name="Remaining",
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        barmode="stack",
        template="simple_white",
        yaxis=dict(range=[0, 100])
    )

    return fig


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
                array=q3 - median,       # upper
                arrayminus=median - q1,  # lower
                thickness=1.5,
                width=4
            ),
            name="Median (Q1–Q3)"
        )
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title="Item",
        yaxis_title="Value",
        showlegend=False,
        height=450
    )
    return fig


def batch_plots(results, title, kind, events, metric):
    df = pd.DataFrame(results).sort_values("step").reset_index(drop=True)

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])
    df["mid_date"]   = df[["start_date", "end_date"]].mean(axis=1)

    # evidence
    evid = df["evidence"].tolist()

    if kind == "binary":
        fig_evi = fig_percentage_bar(evid)
    elif kind == "continuous":
        fig_evi = fig_iqr(np.asarray(evid, dtype=float))
    else:
        raise ValueError("kind must be 'binary' or 'continuous'")

    # ---- Row 1: metric vs mid_date ----
    fig_dist = px.line(
        df,
        x="mid_date",
        y=metric,
        title=title,
        labels={metric: kind.title() + " Distance", "mid_date": "Batch midpoint date"},
        range_y=[0, 1]
    )

    # ---- Row 2: start_date vs mid_date (optional but keeps your “timeline” idea) ----
    # This shows how the batch windows move over time; you could also plot end_date.
    fig_dates = px.scatter(
        df,
        x="mid_date",
        y="start_date",
        title="",
        labels={"mid_date": "Batch midpoint date", "start_date": "Start date"}
    )
    fig_dates.update_traces(mode="lines+markers")

    # ---- Row 3: evidence vs mid_date ----
    # Your fig_percentage_bar / fig_iqr currently creates its own x as 1..n,
    # so we overwrite x to be mid_date for all its traces:
    for tr in fig_evi.data:
        tr.x = df["mid_date"]


    n_rows=3
    plot_height=[0.4, 0.3, 0.3]
    if 'cusum' in list(results[0].keys()):
        # ---- Row 4: cusum ----
        fig_cusum = px.line(
            df,
            x="mid_date",
            y='cusum',
            title=title,
            labels={metric: "cusum", "mid_date": "Batch midpoint date"},
            #range_y=[0, 1]
        )      
        fig_alert = px.line(
            df,
            x="mid_date",
            y='threshold',
            title=title,
            labels={metric: "alert", "mid_date": "Batch midpoint date"},
            #range_y=[0, 1]
        )    
        fig_alert.update_traces(line=dict(color="red"))
            # ---- Combine ----
        n_rows=4
        plot_height=[0.25, 0.25, 0.25,0.25]

        # ---- Combine ----
    combo = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=plot_height,
        subplot_titles=(fig_dist.layout.title.text, "", "")
    )

    for tr in fig_dist.data:
        combo.add_trace(tr, row=1, col=1)

    for tr in fig_dates.data:
        combo.add_trace(tr, row=2, col=1)

    for tr in fig_evi.data:
        combo.add_trace(tr, row=3, col=1)

    if 'cusum' in list(results[0].keys()):
        for tr in fig_cusum.data:
            combo.add_trace(tr, row=4, col=1)
        for tr in fig_alert.data:
            combo.add_trace(tr, row=4, col=1)
        


    combo.update_yaxes(title_text="Distance", range=[0, 1], row=1, col=1)
    combo.update_yaxes(title_text="Start date", row=2, col=1)
    combo.update_xaxes(title_text="Batch midpoint date", row=3, col=1)
    combo.update_yaxes(autorange="reversed", row=2, col=1)

    combo.update_layout(
        template="simple_white",
        height=700,
        showlegend=False,
        margin=dict(t=60, b=40, l=50, r=20)
    )
    
    combo.update_layout(barmode="stack")


    # ---- Events aligned to mid_date ----
    if events:
        events_df = pd.DataFrame(
            [(pd.to_datetime(k), v) for k, v in events.items()],
            columns=["event_date", "label"]
        )

        event_points = []
        for _, r in events_df.iterrows():
            idx = (df["mid_date"] - r["event_date"]).abs().idxmin()
            event_points.append({
                "mid_date": df.loc[idx, "mid_date"],
                "start_date": df.loc[idx, "start_date"],
                "label": r["label"]
            })

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


def weekly_plot(results,title,events,metric):
    # figure
    res_df = pd.DataFrame(results)
    x_vals = pd.to_datetime(res_df["week_start"])

    fig = go.Figure()

    # Distance line (left axis)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=res_df[metric],
            mode="lines+markers",
            name="Distance",
            yaxis="y1"
        )
    )

    # Weekly patient counts (right axis)
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=res_df["n_cur"],   # patients in current week
            name="Weekly patients",
            yaxis="y2",
            opacity=0.5
        )
    )

    # Layout with dual y-axes
    fig.update_layout(
        title=title,
        xaxis=dict(title="Week start"),
        yaxis=dict(title="Distance", range=[0, 1]),   # left
        yaxis2=dict(
            title="Weekly patients",
            overlaying="y",
            side="right",
            showgrid=False
        ),
    
        legend=dict(x=0.01, y=0.99, bordercolor="gray", borderwidth=1)
    )


    # Convert to datetime
    events = {pd.to_datetime(k): v for k, v in events.items()}
    yvals = [0.5 + 0.1 * (-1)**i for i in range(len(events))]
    

    for i, (date, label) in enumerate(events.items()):
        fig.add_trace(
            go.Scatter(
                x=[date],                 # one date
                y=[yvals[i]],             # matching y value
                mode="markers+text",
                text=[label],             # one label
                textposition="top center",
                marker=dict(color="red", size=10, symbol="diamond"),
                showlegend=False,         # no legend
                yaxis="y1"
            )
        )

    fig.update_layout(showlegend=False)  
    return fig      

def heatmap(df_metric):
    # Assuming df_metric is in the format you showed (rows = variables, cols = steps)
    fig = px.imshow(
        df_metric,
        labels=dict(x="Step", y="Variable", color="Metric value"),
        x=df_metric.columns,
        y=df_metric.index,
        aspect="auto",  # so it doesn’t force square cells
        color_continuous_scale="Viridis"  # you can try 'Plasma', 'Cividis', 'Blues' etc.
    )

    fig.update_layout(
        title="Heatmap of Metrics",
        xaxis_title="Step",
        yaxis_title="Variable"
    )

    return fig

def prepare_data(data, value_col, date_col,batch):
    df = data[[value_col, date_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col])             # must have valid dates
    df = df.sort_values(date_col).reset_index(drop=True)
    # convert to float for consistent numeric ops
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    x = df[value_col].to_numpy(dtype=float)
    binary = set(pd.Series(x).dropna().unique()).issubset({0.0, 1.0})

    n = len(x)
    if n < 2 * batch:
        raise ValueError("Need at least 2 batches to compare.")
    
    # group by week
    freq='W-MON'
    df["week"] = df[date_col].dt.to_period(freq)
    weeks = np.sort(df["week"].dropna().unique())  # PeriodIndex sorted

    if len(weeks) < 2:
        raise ValueError("Need data spanning at least 2 distinct weeks.")
    return df,x,n,binary,weeks   

def gen_binary(n: int, p: float, seed: int | None = None) -> np.ndarray:
    """
    Generate binary 0/1 data with Bernoulli(p).
    """
    rng = np.random.default_rng(seed)
    p = float(p)
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    return rng.binomial(n=1, p=p, size=n)


def gen_continuous(
    n: int,
    dist: str = "uniform",
    low: float = 0,
    high: float = 100,
    mean: float = 50,
    sd: float = 10,
    integer: bool = True,
    decimals: int = 1,
    clip: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate continuous (or integer) data within [low, high].
    Supported dist: uniform, normal, triangular
    """
    rng = np.random.default_rng(seed)
    dist = dist.lower()

    if low >= high:
        raise ValueError("low must be < high.")

    if dist == "uniform":
        x = rng.uniform(low, high, size=n)

    elif dist == "normal":
        x = rng.normal(mean, sd, size=n)
        if clip:
            x = np.clip(x, low, high)
        else:
            # still enforce bounds if user requested strict range
            pass

    elif dist == "triangular":
        # mode at mean by default (clipped into [low, high])
        mode = float(np.clip(mean, low, high))
        x = rng.triangular(left=low, mode=mode, right=high, size=n)

    else:
        raise ValueError("dist must be one of: 'uniform', 'normal', 'triangular'.")

    if integer:
        x = np.rint(x).astype(int)
    else:
        x = np.round(x.astype(float), decimals)

    return x


def _date_weights(
    dates: pd.DatetimeIndex,
    pattern: str = "uniform",
    seed: int | None = None,
) -> np.ndarray:
    """
    Create per-day sampling weights for admissions.
    pattern options:
      - uniform: equal probability each day
      - weekday_bias: more admissions on weekdays than weekends
      - weekly_seasonality: smooth weekly wave
      - spiky: mostly low but a few random surge days
    """
    rng = np.random.default_rng(seed)
    pattern = pattern.lower()

    n = len(dates)
    w = np.ones(n, dtype=float)

    if pattern == "uniform":
        w[:] = 1.0

    elif pattern == "weekday_bias":
        # Mon-Fri higher than Sat/Sun
        dow = dates.dayofweek  # 0=Mon ... 6=Sun
        w = np.where(dow < 5, 1.5, 0.6).astype(float)

    elif pattern == "weekly_seasonality":
        # sinusoidal weekly cycle
        t = np.arange(n)
        w = 1.0 + 0.4 * np.sin(2 * np.pi * t / 7.0)

    elif pattern == "spiky":
        # baseline + a few spikes
        w[:] = 0.8
        spike_days = rng.choice(np.arange(n), size=max(1, n // 10), replace=False)
        w[spike_days] = rng.uniform(2.0, 5.0, size=len(spike_days))

    else:
        raise ValueError(
            "pattern must be one of: 'uniform', 'weekday_bias', 'weekly_seasonality', 'spiky'."
        )

    # Normalize to probabilities
    w = np.clip(w, 0, None)
    w = w / w.sum()
    return w


def gen_cohort(
    n_patients: int,
    start_date: str,
    end_date: str,
    cough_p: float = 0.3,
    age_dist: str = "normal",
    age_low: float = 0,
    age_high: float = 100,
    age_mean: float = 45,
    age_sd: float = 18,
    age_integer: bool = True,
    age_decimals: int = 1,
    admissions_pattern: str = "weekday_bias",
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Generate a dummy dataframe with:
      - date_admit (randomly distributed across date range)
      - symptoms_cough (0/1)
      - age (years)

    Admissions distribution is controlled via admissions_pattern.
    """
    rng = np.random.default_rng(seed)

    # Date range (inclusive)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        raise ValueError("Date range produced no dates. Check start/end.")

    # Sample admission dates with weights (this defines "distribution of patients over time")
    probs = _date_weights(dates, pattern=admissions_pattern, seed=seed)
    date_admit = rng.choice(dates.to_numpy(), size=n_patients, replace=True, p=probs)
    date_admit = pd.to_datetime(date_admit)

    # Generate age + cough
    age = gen_continuous(
        n=n_patients,
        dist=age_dist,
        low=age_low,
        high=age_high,
        mean=age_mean,
        sd=age_sd,
        integer=age_integer,
        decimals=age_decimals,
        seed=None if seed is None else seed + 1,
    )

    symptoms_cough = gen_binary(
        n=n_patients,
        p=cough_p,
        seed=None if seed is None else seed + 2,
    )

    df = pd.DataFrame(
        {
            "date_admit": date_admit,
            "age": age,
            "symptoms_cough": symptoms_cough,
        }
    ).sort_values("date_admit").reset_index(drop=True)

    return df



def binary_column(
    n_patients: int,
    start_date: str,
    end_date: str,
    cough_p: float = 0.3,
    admissions_pattern: str = "weekday_bias",
    seed: int | None = 42,
    col_name='binary_col',
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # Date range (inclusive)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        raise ValueError("Date range produced no dates. Check start/end.")

    # Sample admission dates with weights (this defines "distribution of patients over time")
    probs = _date_weights(dates, pattern=admissions_pattern, seed=seed)
    date_admit = rng.choice(dates.to_numpy(), size=n_patients, replace=True, p=probs)
    date_admit = pd.to_datetime(date_admit)

    binary_col = gen_binary(
        n=n_patients,
        p=cough_p,
        seed=None if seed is None else seed + 2,
    )

    df = pd.DataFrame(
        {
            "date_admit": date_admit,
            col_name: binary_col,
        }
    ).sort_values("date_admit").reset_index(drop=True)

    return df

def continuous_column(
    n_patients: int,
    start_date: str,
    end_date: str,
    dist: str = "normal",
    low: float = 0,
    high: float = 100,
    mean: float = 45,
    sd: float = 18,
    col_integer: bool = True,
    col_decimals: int = 1,
    admissions_pattern: str = "weekday_bias",
    seed: int | None = 42,
    col_name='continuous_col',
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # Date range (inclusive)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        raise ValueError("Date range produced no dates. Check start/end.")

    # Sample admission dates with weights (this defines "distribution of patients over time")
    probs = _date_weights(dates, pattern=admissions_pattern, seed=seed)
    date_admit = rng.choice(dates.to_numpy(), size=n_patients, replace=True, p=probs)
    date_admit = pd.to_datetime(date_admit)

    # Generate age + cough
    cont = gen_continuous(
        n=n_patients,
        dist=dist,
        low=low,
        high=high,
        mean=mean,
        sd=sd,
        integer=col_integer,
        decimals=col_decimals,
        seed=None if seed is None else seed + 1,
    )


    df = pd.DataFrame(
        {
            "date_admit": date_admit,
            col_name: cont,
        }
    ).sort_values("date_admit").reset_index(drop=True)

    return df
def rolling_metric_fixed_baseline_cusum(
    data,
    value_col,
    date_col,
    events,
    batch=100,
    baseline_batches=1,     # fixed baseline = first baseline_batches * batch patients
    n_bins=10,
    k=0.0,                 # "noise floor" for the metric (distance) to ignore small/random changes
    metric_name=None,      # if None, uses the first metric returned by *_metrics
    plot=True,
    th=0.5,

    stable_patience=3,
    subsample_rate_low=0.2
):
    """
    Fixed-baseline rolling metric + CUSUM accumulation (no threshold/alarm yet).

    For each batch k (after baseline), compute metric between:
      ref = baseline (fixed)
      cur = current batch

    Then compute:
      excess = metric - k
      cusum  = max(0, prev_cusum + excess)

    Returns:
      results (list of dicts), combo plot (if plot=True)
    """

    # Prepare data (your helper)
    df, x, n, binary, weeks = prepare_data(data, value_col, date_col, batch)

    # Define fixed baseline
    baseline_n = baseline_batches * batch
    if n < baseline_n + batch:
        raise ValueError(
            f"Not enough data for baseline ({baseline_n}) + at least 1 batch ({batch}). "
            f"Got n={n}."
        )

    baseline = x[:baseline_n]

    results = []
    cusum = 0.0

    # number of full batches after baseline
    n_after = n - baseline_n
    n_batches = n_after // batch

    chosen_metric = None
    sampling_rate=1
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
            keep_n = int(np.floor(n_cur * sampling_rate))
            idx = rng.choice(n_cur, size=keep_n, replace=False)
            cur = cur[idx]
        

        # --- REBASELINE MODE: collect baseline_batches full batches ---
        if rebaseline:
            rebaseline_buf.append(cur)  # IMPORTANT: full batch, no subsampling here

            # store a row if you want, or just skip metrics while rebaselining
            if len(rebaseline_buf) < baseline_batches:
                continue  # keep collecting; don't compute metrics/cusum yet

            # we have enough: build new baseline and reset
            baseline = np.concatenate(rebaseline_buf)
            cusum = 0.0
            rebaseline = False
            rebaseline_buf = []

            # after resetting, skip metric computation for this iteration
            # (because this batch was used to build the baseline)
            continue



        cur_dates = df[date_col].iloc[start:end]
        evidence_used = np.concatenate([ref, cur])

        if binary:
            metrics = binary_metrics(ref, cur)
            kind = "binary"
            title = "Fixed Baseline Drift (Binary)"
            n_pos = np.nansum(cur)
            count = int(n_pos)
            pct = (n_pos / len(cur) * 100) if len(cur) > 0 else None
            evidence = [count, pct]
        else:
            metrics = continuous_metrics(ref, cur, n_bins=n_bins)
            kind = "continuous"
            title = "Fixed Baseline Drift (Continuous)"
            med = np.nanmedian(cur)
            q1 = np.nanpercentile(cur, 25)
            q3 = np.nanpercentile(cur, 75)
            evidence = [med, q1, q3]

        # choose which metric to CUSUM
        metric_keys = list(metrics.keys())
        if chosen_metric is None:
            chosen_metric = metric_name if metric_name is not None else metric_keys[0]
            if chosen_metric not in metrics:
                raise ValueError(f"metric_name='{chosen_metric}' not found. Available: {metric_keys}")

        metric_value = metrics.get(chosen_metric, np.nan)



        # CUSUM accumulation
        excess = metric_value - k if (metric_value is not None and not np.isnan(metric_value)) else np.nan
        if not np.isnan(excess):
            cusum = max(0.0, cusum + excess)

        prev_cusum = results[-1]["cusum"] if results else 0.0
        in_alert = int(cusum >= th)
        alert = int((prev_cusum < th) and (cusum >= th))  # one-time trigger

        row = {
            "step": i + 1,  # step 1 = first batch after baseline
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
            "alert":alert,
            "in_alert":in_alert,
            "threshold":th

        }

        # include all metrics in the row (like your style, but simpler)
        row.update(metrics)

        results.append(row)

        if in_alert == 1:
            #print("rebaseline")
            rebaseline=True
            sampling_rate = 1.0
            rebaseline_buf = [] 

        elif len(results) >= stable_patience and all(
                r["in_alert"] == 0 for r in results[-stable_patience:]
            ):
            #print("stable")
            rebaseline=False
            sampling_rate = subsample_rate_low

        else:
            rebaseline=False
            sampling_rate = 1.0




    if plot:
        # You can keep your existing plotting; it expects a metric column name.
        # Here we plot the raw metric (not cusum) unless you extend batch_plots later.
        combo = batch_plots(results, title, kind, events, chosen_metric)
    else:
        combo = []

    return results, combo




def generate_scenario(
    kind: str,                         # "continuous" or "binary"
    scenario_type: str,                # "gradual_drift" | "step_change" | "transient_peak" | "high_noise"
    start_date: str,
    end_date: str,
    events: dict | None,               # {"YYYY-MM-DD": "Label", ...} boundary dates

    # --- NEW: user-defined sample size per block ---
    n_per_block: int = 400,            # used if n_per_block_list is None
    n_per_block_list: list[int] | None = None,

    admissions_pattern: str = "spiky",
    seed_start: int = 100,
    col_name: str = "var",

    # Parameter schedules
    values: list[float] | None = None,     # mean schedule (cont) or p schedule (bin)
    sds: list[float] | None = None,        # optional sd schedule (cont), used for high_noise

    # Continuous defaults
    dist: str = "normal",
    low: float = 0,
    high: float = 95,
    mean: float = 60,
    sd: float = 25,
    col_integer: bool = True,
    col_decimals: int = 1,

    # Binary defaults
    p: float = 0.30,
):
    kind = kind.lower()
    scenario_type = scenario_type.lower()
    if kind not in {"continuous", "binary"}:
        raise ValueError("kind must be 'continuous' or 'binary'")
    if scenario_type not in {"gradual_drift", "step_change", "transient_peak", "high_noise"}:
        raise ValueError("scenario_type must be one of: gradual_drift, step_change, transient_peak, high_noise")

    # ---- build segment boundaries from events ----
    if events is None:
        event_dates = []
    else:
        event_dates = sorted(pd.to_datetime(list(events.keys())))

    boundaries = [pd.to_datetime(start_date)] + event_dates + [pd.to_datetime(end_date)]

    # convert boundaries to inclusive blocks (start..end)
    blocks = []
    for i in range(len(boundaries) - 1):
        b_start = boundaries[i]
        if i < len(boundaries) - 2:
            b_end = boundaries[i + 1] - pd.Timedelta(days=1)
        else:
            b_end = boundaries[i + 1]
        blocks.append((str(b_start.date()), str(b_end.date())))

    n_blocks = len(blocks)

    # ---- NEW: determine n per block ----
    if n_per_block_list is not None:
        if len(n_per_block_list) != n_blocks:
            raise ValueError(
                f"`n_per_block_list` length ({len(n_per_block_list)}) must match number of blocks ({n_blocks})."
            )
        ns = list(n_per_block_list)
    else:
        ns = [int(n_per_block)] * n_blocks

    # ---- default schedules if not provided ----
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

    # ---- generate per-block data ----
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

    return (
        pd.concat(dfs, ignore_index=True)
        .sort_values("date_admit")
        .reset_index(drop=True)
    )


