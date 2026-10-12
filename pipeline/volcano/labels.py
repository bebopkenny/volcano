"""Daily eruption labels."""
import pandas as pd


def label_days(dates: pd.DatetimeIndex, onset: str | None, horizon_days: int):
    """Return (label, pre_onset) for each UTC day.

    A day's features use data up to the end of that day, so the forecast is made
    at midnight UTC. label is 1 when the onset falls within the next `horizon_days`
    after that moment. pre_onset is False for the onset day and everything after;
    those days are dropped from training and scoring because we forecast the start
    of an eruption, not ongoing activity. Unrest cases (onset None) are all 0.
    """
    day_end = dates + pd.Timedelta(days=1)
    if onset is None:
        return pd.Series(0, index=dates), pd.Series(True, index=dates)
    t = pd.Timestamp(onset).tz_convert("UTC").tz_localize(None)
    pre = day_end <= t
    label = pre & (t - day_end <= pd.Timedelta(days=horizon_days))
    return pd.Series(label.astype(int), index=dates), pd.Series(pre, index=dates)
