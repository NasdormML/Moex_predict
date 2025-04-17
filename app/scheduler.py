from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import os
from datetime import datetime, timedelta
import pandas as pd
from app.data import fetch_moex_eod_data

def update_real_prices():
    """
    Фоновая загрузка "истинной" цены закрытия после завершения торгов.
    """
    for ticker in ["SBER", "GAZP"]:
        dt = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        df = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", dt, dt)
        if df is None or df.empty:
            continue
        df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df["TRADEDATE"])).dt.normalize()
        if "CLOSE" not in df.columns:
            continue
        out = df[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": "close"})
        path = "history"
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"real_{ticker}.csv")
        out.to_csv(fn, mode='a' if os.path.exists(fn) else 'w',
                   header=not os.path.exists(fn), index=False)
        print(f"[{datetime.now()}] Saved real {ticker} {dt}")

scheduler = BackgroundScheduler()
scheduler.add_job(update_real_prices, 'cron', hour=19, minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())
