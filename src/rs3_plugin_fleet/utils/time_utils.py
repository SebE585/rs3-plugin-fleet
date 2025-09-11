# src/rs3_plugin_fleet/utils/time_utils.py
from __future__ import annotations
import pandas as pd

EPOCH_MIN = pd.Timestamp("2000-01-01T00:00:00Z")

def parse_start_at(val) -> pd.Timestamp:
    if isinstance(val, pd.Timestamp):
        return val.tz_convert("UTC") if val.tzinfo else val.tz_localize("UTC")
    ts = pd.to_datetime(val, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"start_at invalide: {val!r}")
    return ts

def resolve_abs_time(df: pd.DataFrame, start_at: pd.Timestamp) -> pd.Series:
    if "ts_ms" in df: return pd.to_datetime(df["ts_ms"], unit="ms", utc=True, errors="coerce")
    if "ts_s"  in df: return pd.to_datetime(df["ts_s"],  unit="s",  utc=True, errors="coerce")
    if "timestamp" in df:
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "t_ms" in df:
        return (start_at + pd.to_timedelta(pd.to_numeric(df["t_ms"], errors="coerce"), unit="ms")).astype("datetime64[ns, UTC]")
    if "t_s" in df:
        return (start_at + pd.to_timedelta(pd.to_numeric(df["t_s"], errors="coerce"), unit="s")).astype("datetime64[ns, UTC]")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

def ensure_time_contracts_ok(result) -> None:
    """Stop dur si des timestamps < 2000-01-01 UTC."""
    df = getattr(result, "df", None)
    if df is None or "timestamp" not in df:
        return
    bad = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") < EPOCH_MIN
    if bool(bad.any()):
        n = int(bad.sum())
        raise AssertionError(f"[TIME] {n} timestamps < {EPOCH_MIN.isoformat()} — probable dérive → stop dur.")