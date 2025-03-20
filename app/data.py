import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def get_with_retries(url, params, timeout=30, retries=3):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print("Ошибка запроса:", e)
        return None

def fetch_moex_intraday_data(security, interval, from_date, till_date, start=0, limit=100):
    # Определяем рынок и доску в зависимости от тикера
    if security.upper() == "SBER":
        market, board = "shares", "TQBR"
    elif security.upper() == "IMOEX":
        market, board = "index", "SNDX"
    else:
        # Для USD и других используем соответствующую категорию
        market, board = "currency", "selt"
    
    base_url = f"https://iss.moex.com/iss/engines/stock/markets/{market}/securities/{security}/candles.json"
    
    all_data = []
    columns = None
    while True:
        params = {
            "interval": interval,
            "from": from_date,
            "till": till_date,
            "iss.meta": "off",
            "start": start
        }
        response = get_with_retries(base_url, params=params, timeout=30, retries=3)
        if response is None:
            break
        data = response.json()
        if "candles" not in data or "columns" not in data["candles"]:
            print(f"Ошибка: ответ для {security} intraday не содержит ожидаемых ключей 'candles'/'columns'")
            return None
        if columns is None:
            columns = data["candles"]["columns"]
        page_data = data["candles"].get("data", [])
        if not page_data:
            break
        all_data.extend(page_data)
        if len(page_data) < limit:
            break
        start += len(page_data)
    if all_data and columns:
        return pd.DataFrame(all_data, columns=columns)
    else:
        return None

def fetch_moex_eod_data(security, engine, market, board, start_date, end_date):
    base_url = f"https://iss.moex.com/iss/history/engines/{engine}/markets/{market}/boards/{board}/securities/{security}.json"
    all_data = []
    columns = None
    offset = 0
    limit = 100

    while True:
        params = {"from": start_date, "till": end_date, "start": offset}
        response = get_with_retries(base_url, params=params, timeout=30, retries=3)
        if response is None:
            break
        data = response.json()
        try:
            if columns is None:
                columns = data["history"]["columns"]
            page_data = data["history"]["data"]
            if not page_data:
                break
            all_data.extend(page_data)
            if len(page_data) < limit:
                break
            offset += limit
        except KeyError:
            print("Ошибка формата данных.")
            break

    if all_data and columns:
        return pd.DataFrame(all_data, columns=columns)
    else:
        return None
