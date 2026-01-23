import pickle
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import redis # type: ignore
from app.data import _session, logger

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 1
REDIS_TTL_SECONDS = 86400

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    """Ленивый singleton Redis-клиент."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
    return _redis_client

def _cache_key(ticker: str, trade_date: date) -> str:
    return f"intraday:1m:{ticker}:{trade_date.isoformat()}"


def _load_cached_redis(ticker: str, trade_date: date) -> Optional[pd.DataFrame]:
    """LRU-cache через Redis, fallback — локальный pickle."""
    r = _get_redis()
    key = _cache_key(ticker, trade_date)
    try:
        raw = r.get(key)
        if raw:
            df = pickle.loads(raw)
            logger.debug("Redis hit: %s", key)
            return df
    except redis.RedisError as exc:
        logger.warning("Redis unavailable: %s", exc)

    # Fallback — локальный pickle
    from app.data import CACHE_DIR
    path = CACHE_DIR / "intraday_1m" / f"{ticker}_{trade_date}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            df = pd.read_pickle(path)
            logger.debug("Pickle hit: %s", path.name)
            return df
        except Exception as exc:
            logger.warning("Broken pickle %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
    return None


def _save_cached(df: pd.DataFrame, ticker: str, trade_date: date) -> None:
    """Сохранение в Redis + pickle fallback."""
    r = _get_redis()
    key = _cache_key(ticker, trade_date)
    try:
        raw = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        r.setex(key, REDIS_TTL_SECONDS, raw)
        logger.info("Cached Redis: %s", key)
    except redis.RedisError as exc:
        logger.warning("Redis save failed: %s", exc)
        # Fallback — pickle
        from app.data import CACHE_DIR
        path = CACHE_DIR / "intraday_1m" / f"{ticker}_{trade_date}.pkl"
        try:
            df.to_pickle(path)
            logger.info("Cached pickle: %s", path.name)
        except Exception as exc2:
            logger.error("Pickle save failed: %s", exc2)

def fetch_minute_bars(
    ticker: str,
    trade_date: date,
    *,
    board: str = "TQBR",
    market: str = "shares",
    engine: str = "stock",
) -> pd.DataFrame:
    cached = _load_cached_redis(ticker, trade_date)
    if cached is not None:
        return cached

    start_dt = pd.Timestamp(trade_date)
    end_dt = start_dt + timedelta(days=1)

    url = (
        f"https://iss.moex.com/iss/engines/{engine}/markets/{market}/"
        f"boards/{board}/securities/{ticker}/candles.json"
    )
    params = {
        "from": start_dt.strftime("%Y-%m-%d"),
        "till": end_dt.strftime("%Y-%m-%d"),
        "interval": 1,
        "start": 0,
        "limit": 50_000,
    }

    all_data: List[List] = []
    session = _session
    while True:
        try:
            resp = session.get(url, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("MOEX 1m fetch failed: %s", exc)
            break

        payload = resp.json()
        candles = payload.get("candles", {})
        data = candles.get("data", [])
        if not data:
            break
        all_data.extend(data)
        if len(data) < params["limit"]:
            break
        params["start"] += params["limit"]

    if not all_data:
        df = pd.DataFrame()
    else:
        columns = candles.get("columns", [])
        df = pd.DataFrame(all_data, columns=columns)
        df = df.rename(columns=str.lower)
        df["TRADEDATE"] = pd.to_datetime(df["begin"]).dt.date
        df["TIME"] = pd.to_datetime(df["begin"]).dt.time
        df = df[["TRADEDATE", "TIME", "open", "high", "low", "close", "volume"]].copy()

    if not df.empty:
        _save_cached(df, ticker, trade_date)
    else:
        logger.warning("No 1m data for %s on %s", ticker, trade_date)
    return df

def aggregate_intraday_features(df_min: pd.DataFrame) -> pd.Series:
    if df_min.empty:
        # Заглушка нулями
        return pd.Series({
            "vwap_intra": 0.0, "volatility_intra": 0.0, "kurtosis_intra": 0.0,
            "skew_intra": 0.0, "buy_vol_ratio": 0.5, "open_30min_ret": 0.0,
            "close_30min_ret": 0.0, "volume_intra": 0, "trades_count": 0,
            "depth_imbalance": 0.0, "large_trade_ratio": 0.0,
            "TRADEDATE": pd.NaT,
        })

    # VWAP
    vwap = (df_min["close"] * df_min["volume"]).sum() / (df_min["volume"].sum() + 1e-8)

    ret_1m = df_min["close"].pct_change().dropna()
    vol_intra = ret_1m.std()
    kurt_intra = ret_1m.kurtosis()
    skew_intra = ret_1m.skew()

    # Buy/Sell imbalance
    buy_vol = df_min.loc[df_min["close"] > df_min["open"], "volume"].sum()
    sell_vol = df_min.loc[df_min["close"] < df_min["open"], "volume"].sum()
    buy_ratio = buy_vol / (buy_vol + sell_vol + 1e-8)

    # Открытие/закрытие дня
    open_30 = df_min.head(30)["close"].pct_change().iloc[-1] if len(df_min) >= 30 else 0.0
    close_30 = df_min.tail(30)["close"].pct_change().iloc[-1] if len(df_min) >= 30 else 0.0

    # top-5% по объёму
    vol_95 = df_min["volume"].quantile(0.95)
    large_ratio = (df_min["volume"] >= vol_95).mean()

    date = df_min["TRADEDATE"].iloc[0] if not df_min.empty else pd.NaT
    return pd.Series({
        "vwap_intra": vwap,
        "volatility_intra": vol_intra,
        "kurtosis_intra": kurt_intra,
        "skew_intra": skew_intra,
        "buy_vol_ratio": buy_ratio,
        "open_30min_ret": open_30,
        "close_30min_ret": close_30,
        "volume_intra": df_min["volume"].sum(),
        "trades_count": len(df_min),
        "depth_imbalance": 0.0,
        "large_trade_ratio": large_ratio,
        "TRADEDATE": date,
    })


def add_intraday_features(
    df_daily: pd.DataFrame,
    ticker: str,
    *,
    board: str = "TQBR",
) -> pd.DataFrame:
    if df_daily.empty or "TRADEDATE" not in df_daily.columns:
        return df_daily

    dates = pd.to_datetime(df_daily["TRADEDATE"]).dt.date.unique()
    features_list: list[pd.Series] = []

    for d in dates:
        df_min = fetch_minute_bars(ticker, d, board=board)
        agg = aggregate_intraday_features(df_min)
        features_list.append(agg)

    if not features_list:
        # Заглушка на случай отсутствия данных
        empty = pd.Series(0.0, index=[
            "vwap_intra", "volatility_intra", "kurtosis_intra", "skew_intra",
            "buy_vol_ratio", "open_30min_ret", "close_30min_ret",
            "volume_intra", "trades_count", "depth_imbalance", "large_trade_ratio"
        ])
        empty["TRADEDATE"] = pd.to_datetime(dates.min()).date()
        features_list = [empty]

    intra_df = pd.DataFrame(features_list)
    return df_daily.merge(intra_df, on="TRADEDATE", how="left")