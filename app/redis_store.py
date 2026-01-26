import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

import pandas as pd
import redis # type: ignore
from redis.exceptions import RedisError # type: ignore

logger = logging.getLogger(__name__)

class MoexRedisStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(
            host=host, 
            port=port, 
            db=db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30
        )
        self.ttl_daily = 86400 * 7
        self.ttl_intraday = 86400 * 2
        self.ttl_forecast = 86400 * 30
        
    def _key(self, *parts) -> str:
        """Generate Redis key: moex:daily:SBER:20250118"""
        return ":".join(str(p) for p in parts)
    
    # ========== MOEX Data Storage ==========
    
    def store_daily_bar(self, ticker: str, date: str, data: Dict[str, Any]):
        """Store single daily OHLCV bar.
        """
        key = self._key("moex", "daily", ticker.upper(), date.replace("-", ""))
        try:
            self.client.hset(key, mapping={
                "open": float(data["open"]),
                "high": float(data["high"]),
                "low": float(data["low"]),
                "close": float(data["close"]),
                "volume": int(data["volume"]),
                "updated_at": datetime.now().isoformat()
            })
            self.client.expire(key, self.ttl_daily)
            logger.debug(f"Redis stored: {key}")
        except RedisError as e:
            logger.error(f"Redis store failed: {e}")
    
    def get_daily_bar(self, ticker: str, date: str) -> Optional[Dict]:
        """Get single daily bar.
        """
        key = self._key("moex", "daily", ticker.upper(), date.replace("-", ""))
        try:
            data = self.client.hgetall(key)
            if not data:
                return None
            return {
                "open": float(data["open"]),
                "high": float(data["high"]),
                "low": float(data["low"]),
                "close": float(data["close"]),
                "volume": int(data["volume"]),
                "date": date
            }
        except RedisError as e:
            logger.error(f"Redis get failed: {e}")
            return None
    
    def store_daily_batch(self, ticker: str, df: pd.DataFrame):
        pipe = self.client.pipeline()
        ticker = ticker.upper()
        
        for _, row in df.iterrows():
            date = pd.to_datetime(row["TRADEDATE"]).strftime("%Y%m%d")
            key = self._key("moex", "daily", ticker, date)
            
            pipe.hset(key, mapping={
                "open": float(row.get("OPEN", row.get("open", 0))),
                "high": float(row.get("HIGH", row.get("high", 0))),
                "low": float(row.get("LOW", row.get("low", 0))),
                "close": float(row.get("CLOSE", row.get("close", 0))),
                "volume": int(row.get("VOLUME", row.get("volume", 0))),
                "updated_at": datetime.now().isoformat()
            })
            pipe.expire(key, self.ttl_daily)
        
        try:
            pipe.execute()
            logger.info(f"Redis stored {len(df)} bars for {ticker}")
        except RedisError as e:
            logger.error(f"Redis batch store failed: {e}")
    
    def get_daily_range(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get range of daily bars as DataFrame."""
        ticker = ticker.upper()
        dates = pd.date_range(start_date, end_date, freq="B")
        
        pipe = self.client.pipeline()
        for date in dates:
            key = self._key("moex", "daily", ticker, date.strftime("%Y%m%d"))
            pipe.hgetall(key)
        
        try:
            results = pipe.execute()
            records = []
            for date, data in zip(dates, results):
                if data:
                    records.append({
                        "TRADEDATE": date,
                        "OPEN": float(data["open"]),
                        "HIGH": float(data["high"]),
                        "LOW": float(data["low"]),
                        "CLOSE": float(data["close"]),
                        "VOLUME": int(data["volume"])
                    })
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
            return df.sort_values("TRADEDATE")
        except RedisError as e:
            logger.error(f"Redis range get failed: {e}")
            return pd.DataFrame()
    
    def store_intraday(self, ticker: str, date: str, df: pd.DataFrame):
        """Store 1-minute bars as JSON list."""
        key = self._key("moex", "intraday", ticker.upper(), date.replace("-", ""))
        try:
            data = df.to_json(orient="records", date_format="iso")
            self.client.setex(key, self.ttl_intraday, data)
            logger.debug(f"Redis stored intraday: {key} ({len(df)} bars)")
        except RedisError as e:
            logger.error(f"Redis intraday store failed: {e}")
    
    def get_intraday(self, ticker: str, date: str) -> Optional[pd.DataFrame]:
        """Get 1-minute bars."""
        key = self._key("moex", "intraday", ticker.upper(), date.replace("-", ""))
        try:
            data = self.client.get(key)
            if not data:
                return None
            return pd.read_json(data, orient="records")
        except RedisError as e:
            logger.error(f"Redis intraday get failed: {e}")
            return None
    
    # ========== Forecasts Storage ==========
    
    def store_forecast(self, ticker: str, forecast_data: Dict, model_version: str):
        """
        Store prediction with full metadata.
        forecast_data: {
            "dates": ["2025-01-20", ...],
            "mean": [307.4, ...],
            "lower": [290.1, ...],      # optional for quantile
            "upper": [325.8, ...],      # optional for quantile
            "input_hash": "sha256..."   # for reproducibility
        }
        """
        ticker = ticker.upper()
        timestamp = datetime.now().isoformat()
        
        #Store as "latest"
        latest_key = self._key("forecast", ticker, "latest")
        pipe = self.client.pipeline()
        
        pipe.hset(latest_key, mapping={
            "timestamp": timestamp,
            "model_version": model_version,
            "horizon": len(forecast_data["mean"]),
            "predictions": json.dumps(forecast_data["mean"]),
            "quantiles_lower": json.dumps(forecast_data.get("lower", [])),
            "quantiles_upper": json.dumps(forecast_data.get("upper", [])),
            "dates": json.dumps(forecast_data["dates"]),
            "input_hash": forecast_data.get("input_hash", ""),
            "confidence_score": forecast_data.get("confidence_score", 0.0)
        })
        pipe.expire(latest_key, self.ttl_forecast)
        
        # Add to history stream
        history_key = self._key("forecast", ticker, "history")
        pipe.xadd(history_key, {
            "timestamp": timestamp,
            "model_version": model_version,
            "predictions_json": json.dumps(forecast_data["mean"]),
            "has_quantiles": "1" if "lower" in forecast_data else "0"
        }, maxlen=1000)
        
        try:
            pipe.execute()
            logger.info(f"Forecast stored in Redis for {ticker}")
        except RedisError as e:
            logger.error(f"Redis forecast store failed: {e}")
    
    def get_latest_forecast(self, ticker: str) -> Optional[Dict]:
        """Get latest prediction with all metadata."""
        key = self._key("forecast", ticker, "latest")
        try:
            data = self.client.hgetall(key)
            if not data:
                return None
            
            return {
                "timestamp": data["timestamp"],
                "model_version": data["model_version"],
                "horizon": int(data["horizon"]),
                "predictions": json.loads(data["predictions"]),
                "quantiles_lower": json.loads(data.get("quantiles_lower", "[]")),
                "quantiles_upper": json.loads(data.get("quantiles_upper", "[]")),
                "dates": json.loads(data["dates"]),
                "confidence_score": float(data.get("confidence_score", 0))
            }
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Redis forecast get failed: {e}")
            return None
    
    def update_actual_price(self, ticker: str, date: str, actual_price: float):
        pass

redis_store = MoexRedisStore()