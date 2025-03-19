import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_candles(security, market, interval=1, days_back=365):
    """Загружает исторические свечные данные"""
    base_url = "https://iss.moex.com/iss/engines/stock/markets/{market}/securities/{security}/candles.json"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    all_data = []
    
    params = {
        "interval": interval,
        "from": start_date.strftime("%Y-%m-%d"),
        "till": end_date.strftime("%Y-%m-%d"),
        "iss.meta": "off",
        "start": 0
    }

    while True:
        try:
            response = requests.get(
                base_url.format(market=market, security=security),
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            candles = data["candles"]["data"]
            
            if not candles:
                break
                
            columns = data["candles"]["columns"]
            df_chunk = pd.DataFrame(candles, columns=columns)
            
            all_data.append(df_chunk)
            
            if len(candles) < 100:
                break
                
            params["start"] += 100
            
        except Exception as e:
            print(f"Ошибка загрузки данных для {security}: {str(e)}")
            break

    if not all_data:
        return None

    df = pd.concat(all_data)
    df.rename(columns={
        "begin": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    }, inplace=True)
    
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date")

def fetch_all_data(ticker_configs):
    """Загружает данные для нескольких тикеров"""
    dataframes = []
    
    for config in ticker_configs:
        df = fetch_candles(
            security=config["security"],
            market=config["market"],
            interval=config.get("interval", 1),
            days_back=config.get("days_back", 30)
        )
        
        if df is not None:
            df = df[["date", "close"]].rename(columns={
                "close": f"close_{config['security']}"
            })
            dataframes.append(df)
    
    if not dataframes:
        return None
    
    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.merge(df, on="date", how="outer")
        
    return merged.sort_values("date").ffill().bfill()