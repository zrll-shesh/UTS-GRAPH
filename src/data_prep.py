"""Loading, filtering and saving the RetailRocket event log."""
from pathlib import Path

import pandas as pd

EVENT_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]


def to_epoch_ms(moment: str) -> int:
    """Convert a UTC date string to a UNIX timestamp in milliseconds (the dataset's unit)."""
    return int(round(pd.Timestamp(moment, tz="UTC").timestamp() * 1000))


def load_events(path: Path) -> pd.DataFrame:
    """Read events.csv with compact dtypes and add a readable UTC datetime column."""
    events = pd.read_csv(
        path,
        usecols=EVENT_COLUMNS,
        dtype={"timestamp": "int64", "visitorid": "int64", "itemid": "int64", "event": "category"},
    )
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    return events


def filter_period(events: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    """Keep events whose timestamp lies inside [start, end] (both inclusive)."""
    start, end = (to_epoch_ms(p) for p in period)
    if start >= end:
        raise ValueError("Period start must be earlier than its end.")
    mask = events["timestamp"].between(start, end)
    return events.loc[mask].sort_values("timestamp", kind="stable").reset_index(drop=True)


def periods_overlap(p1: tuple[str, str], p2: tuple[str, str]) -> bool:
    """True when the two periods share any moment (they must not for a clean split)."""
    (s1, e1), (s2, e2) = ([to_epoch_ms(x) for x in p] for p in (p1, p2))
    return s1 <= e2 and s2 <= e1


def save_events(events: pd.DataFrame, path: Path) -> Path:
    """Write the original five columns; transactionid is stored as a nullable integer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = events[EVENT_COLUMNS].copy()
    out["transactionid"] = out["transactionid"].astype("Int64")
    out.to_csv(path, index=False)
    return path
