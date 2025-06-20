from unittest import mock

import pandas as pd

from app.data import fetch_moex_eod_data, fetch_usd_series, get_dataloaders


@mock.patch("app.data.session.get")
def test_fetch_moex_eod_data_success(mock_get):
    # Мокаем JSON-ответ от MOEX ISS API
    mock_json = {
        "history": {
            "data": [["2025-01-01", 100, 110, 90, 105, 1000]],
            "columns": ["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"],
        }
    }
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_json

    df = fetch_moex_eod_data(
        "SBER", "stock", "shares", "TQBR", "2025-01-01", "2025-01-02", skip_cache=True
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == mock_json["history"]["columns"]


@mock.patch("app.data.fetch_cbr_usd_rate", return_value=60.0)
def test_fetch_usd_series(mock_rate):
    df = fetch_usd_series("2025-01-01", "2025-01-03", skip_cache=True)
    assert isinstance(df, pd.DataFrame)
    # Даты идут последовательно и есть столбец CLOSE
    assert list(df["CLOSE"]) == [60.0, 60.0, 60.0]


@mock.patch("app.data.fetch_moex_eod_data")
@mock.patch("app.data.fetch_usd_series")
def test_get_dataloaders_shape(mock_usd, mock_moex):
    # Мокаем данные для всех трёх источников внутри get_dataloaders
    dates = pd.date_range("2025-01-01", periods=30)
    df_base = pd.DataFrame({"TRADEDATE": dates, "CLOSE": range(30)})
    mock_moex.return_value = df_base.copy()
    mock_usd.return_value = df_base.copy()

    train_dl, val_dl = get_dataloaders(
        "SBER", batch_size=8, start_date="2025-01-01", end_date="2025-01-30"
    )
    batch_x, batch_y = next(iter(train_dl))
    # Проверяем размерности: seq_len=20, batch_size≤8
    assert batch_x.shape[1] == 20
    assert batch_x.shape[0] <= 8
    assert batch_y.shape[0] == batch_x.shape[0]
