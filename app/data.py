import requests
import pandas as pd
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime

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


def fetch_moex_eod_data(security, engine, market, board, start_date, end_date):
    base_url = f"https://iss.moex.com/iss/history/engines/{engine}/markets/{market}/boards/{board}/securities/{security}.json"
    all_data, columns, offset, limit = [], None, 0, 100
    while True:
        params = {"from": start_date, "till": end_date, "start": offset, "limit": limit}
        resp = get_with_retries(base_url, params)
        if not resp:
            break
        data = resp.json()
        if columns is None:
            columns = data.get("history", {}).get("columns")
        page = data.get("history", {}).get("data", [])
        if not page:
            break
        all_data.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return pd.DataFrame(all_data, columns=columns) if all_data else None


def fetch_cbr_usd_rate(date_obj: datetime) -> float:
    date_str = date_obj.strftime("%d/%m/%Y")
    url = f"http://www.cbr.ru/scripts/XML_daily.asp?date_req={date_str}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for v in root.findall('Valute'):
            if v.find('CharCode').text == 'USD':
                return float(v.find('Value').text.replace(',', '.'))
    except Exception:
        pass
    return None
