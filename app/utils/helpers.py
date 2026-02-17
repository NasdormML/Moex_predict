"""Common utilities and helpers"""

import asyncio
from datetime import date, datetime, timedelta

import pandas as pd
from fastapi import HTTPException


def validate_ticker(t: str) -> str:
    """Validate and normalize ticker symbol."""
    t = t.upper().strip()
    if not t.isalnum() or len(t) > 10:
        raise HTTPException(400, f"Invalid ticker: {t}")
    return t


def get_last_known_date(metadata: dict) -> date:
    """Extract last known date from metadata."""
    data_upto = metadata.get("data_upto")
    if data_upto:
        try:
            return datetime.strptime(data_upto, "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.today() - timedelta(days=1)


async def _maybe_async_call(func, *args, **kwargs):
    """Call function - await if coroutine, else run in thread."""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def business_days_between(start: date, end: date) -> int:
    """Count business days between dates (exclusive of start)."""
    if end <= start:
        return 0
    import numpy as np

    return np.busday_count(start.isoformat(), end.isoformat())


def merge_dataframes(df_t, df_i, df_u, ticker: str) -> pd.DataFrame:
    """Merge ticker, index and forex data."""

    def prepare(df, rename_map, columns):
        if df is None or df.empty:
            raise HTTPException(503, "Empty dataframe")

        df = df.copy()
        date_col = "BEGIN" if "BEGIN" in df.columns else "TRADEDATE"
        if date_col not in df.columns:
            raise HTTPException(500, "No date column")

        df["TRADEDATE"] = pd.to_datetime(df[date_col]).dt.normalize()
        df = df.rename(columns=rename_map)

        missing = set(columns) - set(df.columns)
        if missing:
            raise HTTPException(500, f"Missing columns: {missing}")

        return df[columns]

    t_cols = [
        "TRADEDATE",
        f"OPEN_{ticker}",
        f"HIGH_{ticker}",
        f"LOW_{ticker}",
        f"CLOSE_{ticker}",
        f"VOL_{ticker}",
    ]

    df_t = prepare(
        df_t,
        {
            "OPEN": f"OPEN_{ticker}",
            "HIGH": f"HIGH_{ticker}",
            "LOW": f"LOW_{ticker}",
            "CLOSE": f"CLOSE_{ticker}",
            "VOLUME": f"VOL_{ticker}",
        },
        t_cols,
    )

    df_i = prepare(df_i, {"CLOSE": "CLOSE_IMOEX"}, ["TRADEDATE", "CLOSE_IMOEX"])
    df_u = prepare(df_u, {"CLOSE": "CLOSE_USD"}, ["TRADEDATE", "CLOSE_USD"])

    merged = (
        df_t.merge(df_i, on="TRADEDATE", how="outer")
        .merge(df_u, on="TRADEDATE", how="outer")
        .sort_values("TRADEDATE")
        .ffill()
        .dropna()
        .reset_index(drop=True)
    )

    return merged
