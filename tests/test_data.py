from unittest import mock

import pandas as pd

from app.data import create_dataloader, fetch_moex_data


@mock.patch("app.data.requests.get")
def test_fetch_moex_data_success(mock_get):
    sample_csv = "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME\n2025-01-01,100,110,90,105,1000"
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = sample_csv

    df = fetch_moex_data("SBER", "2025-01-01", "2025-01-02")
    assert isinstance(df, pd.DataFrame)
    assert "CLOSE" in df.columns


def test_create_dataloader_shape():
    df = pd.DataFrame({"CLOSE": range(100)})
    dl = create_dataloader(df, window_size=10, batch_size=8)
    batch = next(iter(dl))
    # batch[0] — тензор признаков, batch[1] — тензор таргетов
    assert batch[0].shape == (8, 10, 1)
    assert batch[1].shape == (8, 1)
