# MOEX Price Prediction

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Hydra](https://img.shields.io/badge/hydra-1.3.2-blue)
![Pandas](https://img.shields.io/badge/pandas-2.2.3-%23150458?logo=pandas\&logoColor=white)
![Numpy](https://img.shields.io/badge/numpy-2.2.4-%23013243?logo=numpy\&logoColor=white)
![PyTorch](https://img.shields.io/badge/torch-2.6.0-%23EE4C2C?logo=pytorch\&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-2.21.3-%23004750?logo=mlflow\&logoColor=white)
![Apscheduler](https://img.shields.io/badge/apscheduler-3.11.0-blue)
![Requests](https://img.shields.io/badge/requests-2.32.3-%23BA1200?logo=requests\&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-0.115.12-%23009ECE?logo=fastapi\&logoColor=white)
![Pydantic](https://img.shields.io/badge/pydantic-2.11.3-%23008BD3?logo=pydantic\&logoColor=white)
![Uvicorn](https://img.shields.io/badge/uvicorn-0.34.1-%232C3E50?logo=uvicorn\&logoColor=white)
![Flake8](https://img.shields.io/badge/code%20style-Flake8-4183C4?logo=flake8&logoColor=white)


**MOEX Price Prediction** is an end-to-end forecasting solution for Moscow Exchange (MOEX) stocks (e.g. SBER, GAZP, ROSN) using historical data, technical indicators, and deep learning with PyTorch.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture & Structure](#architecture--structure)
4. [Technologies](#technologies)
5. [Hydra Configuration](#hydra-configuration)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [API Reference](#api-reference)
9. [Model Training & Visualization](#model-training--visualization)
10. [License](#license)

---

## Project Overview

This project fetches MOEX historical end‑of‑day data (for **any ticker** available via the MOEX API in `app/data.py`), computes technical indicators (RSI, SMA, MACD, Bollinger Bands, ATR), trains PyTorch models (LSTM‑Attention, TCN, Transformer) via Hydra‑powered experiments, and serves predictions through a FastAPI REST API. Check [Hydra Configuration](#hydra-configuration).

---

## Key Features

* **Modular Architecture:** Clear separation between data ingestion, preprocessing, model training, and serving.
* **Hydra Experiments:** Easily switch models (`lstm`, `tcn`, `tft`) and tickers (`SBER`, `GAZP`, `ROSN`, etc.) with command-line overrides.
* **Versioned Artifacts:** Models and scalers saved under `saved_models/v{version}`; metadata tracks versions and architecture.
* **Auto‑Retraining:** Optional performance monitoring triggers retraining via MLflow and Optuna.
* **Live API:** FastAPI endpoint for on‑demand predictions.

---

## Architecture & Structure

```plaintext
MOEX_PREDICT/
├── app/
│   ├── data.py                 # MOEX & CBR data loader + DataLoader factory
│   ├── preprocessing.py        # Indicator calculations
│   ├── models/                 # Model definitions and factory
│   │   ├── attention_lstm.py
│   │   ├── tcn.py
│   │   ├── tft.py
│   │   └── factory.py
│   ├── model_manager.py       # Loading versioned models & scalers
│   ├── transfer_learning.py   # Retraining logic & metadata
│   ├── predict.py             # Prediction wrapper
│   ├── monitoring.py          # Performance validation
│   └── main.py                # FastAPI application
├── conf/                      # Hydra configuration
│   ├── config.yaml
│   ├── data/
│   │   └── default.yaml       # Data loader settings
│   ├── model/
│   │   ├── lstm.yaml          # LSTM params
│   │   ├── tcn.yaml           # TCN params
│   │   └── tft.yaml           # Transformer params
│   ├── optimization
│   │   └── default.yaml       # HPO settings
│   │
│   └── train/
│       └── default.yaml       # Training settings & versioning
├── saved_models/              # Versioned model artifacts
│   ├── v1/
│   │   ├── SBER_model.pth
│   │   ├── SBER_scaler_X.pkl
│   │   ├── SBER_scaler_y.pkl
│   │   └── ...
│   └── v2/ ...
├── train.py                   # Hydra entrypoint for experiments
├── Makefile                   # install, run api, mlflow, clean
└── README.md                  # This file
```

---

## Technologies

* **Python:** 3.11
* **Config:** Hydra (1.3.2), OmegaConf
* **Data:** Pandas, NumPy, scikit-learn
* **Deep Learning:** PyTorch 2.6
* **Experiment Tracking:** MLflow 2.21
* **Web API:** FastAPI, Uvicorn
* **Scheduling:** APScheduler

---

## Hydra Configuration

All experiments are driven by `conf/config.yaml`. Override sections via CLI:

```bash
# Train TCN on GAZP, version v2:
python3 train.py \
   model=tcn \
   data.ticker=GAZP \
   train.horizon=5 \
   train.version=v2

# Train model with HPO:
 python3 train.py \
  model=lstm \
  data.ticker=SBER \
  data.start_date=2013-01-01 \
  train.horizon=5 \
  train.epochs=20 \
  train.version=v1 \
  optimization.enable=true \
  optimization.n_trials=20 \
  optimization.epochs_per_trial=20
```

See `conf/` for defaults.

---

## Installation & Setup

```bash
# 1. Clone
git clone https://github.com/NasdormML/Moex_predict.git
cd Moex_predict

# 2. Create Virtual env & Install deps
make install

# 3. Run MLflow & FastAPI
make run
```

* **MLflow UI:** [http://127.0.0.1:5001](http://127.0.0.1:5001)
* **API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Usage

1. **Train a model:**

   ```bash
   python train.py model=lstm data.ticker=SBER
   ```

2. **Serve predictions:**

   ```bash
   make api
   curl -X POST "http://127.0.0.1:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"ticker":"SBER","start_date":"2025-01-01","end_date":"2025-05-01"}'
   ```

---

## API Reference

**POST /predict**
```bash
{
  "ticker": "SBER",
  "known_up_to": "2025-06-28",
  "forecast_dates": [
    "2025-06-30",
    "2025-07-01",
    "2025-07-02",
    "2025-07-03",
    "2025-07-04"
  ],
  "predictions": [
    307.3966369628906,
    306.07830810546875,
    305.132080078125,
    305.0485534667969,
    305.2849426269531
  ]
}
  ```

---

## Model Training & Visualization

<img src="https://github.com/user-attachments/assets/e1af2d59-fa59-405b-a55c-6bd60e48756b">

**SBER Performance**

| Model       | MSE   | RMSE | MAE  | MAPE  |
| ----------- | ----- | ---- | ---- | ----- |
| LSTM-Attn   | 19.87 | 4.46 | 3.11 | 1.20% |
| Transformer | 11.37 | 3.37 | 2.42 | 0.97% |

<img src="https://github.com/user-attachments/assets/060b4922-4ba7-4bad-9bef-d9e25f047030">

**GAZP Performance**

| Model | MSE   | RMSE | MAE  | MAPE  |
| ----- | ----- | ---- | ---- | ----- |
| TCN   | 18.31 | 4.28 | 3.04 | 2.08% |

<img src="https://github.com/user-attachments/assets/1f8ccaa2-942e-4810-81c4-ae0fe939f349">

**ROSN Performance**

| Model       | MSE   | RMSE | MAE  | MAPE  |
| ----------- | ----- | ---- | ---- | ----- |
| Transformer | 61.50 | 7.84 | 5.65 | 1.10% |

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).
