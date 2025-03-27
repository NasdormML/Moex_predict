## MOEX Price Prediction

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![FastAPI 0.95+](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)

**MOEX Price Prediction** is an end-to-end project designed to forecast Moscow Exchange (MOEX) stock prices (e.g., SBER) using historical data, technical indicators (RSI, SMA), and a deep learning model (LSTM with an Attention mechanism). The predictions are served via a REST API built with FastAPI.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Key Components](#key-components)
4. [Installation](#installation)
5. [Model Training](#model-training)
6. [Running the FastAPI App](#running-the-fastapi-app)
7. [API Usage](#api-usage)
8. [Notebooks](#notebooks)
9. [License](#license)

---

## Overview

- **Data Source:**  
  Fetches data from the MOEX ISS API for stocks (`TQBR`), indices (`SNDX` for IMOEX), and currency pairs (`CETS` for USD/RUB).  
  Missing USD data can be replaced by the Central Bank of Russia (CBR) rates if needed.

- **Model Architecture:**  
  Implements an **LSTM** network with an **Attention** layer to handle time-series patterns and focus on the most relevant timesteps.

- **Technical Indicators:**  
  - **RSI (Relative Strength Index)**  
  - **SMA (Simple Moving Average)**  

- **Deployment:**  
  The trained model is served via a **FastAPI** endpoint for real-time predictions.

---

## Project Structure


```plaintext
MOEX_PREDICT/
├── app/
│   ├── __init__.py
│   ├── data.py            # Data fetching from MOEX & CBR
│   ├── main.py            # FastAPI entry point
│   ├── model_manager.py   # Model/scaler loading utilities
│   ├── models.py          # PyTorch model (LSTM + Attention)
│   ├── optimization.py    # (Optional) Hyperparameter optimization
│   ├── predict.py         # Prediction logic using the trained model
│   ├── preprocessing.py   # Preprocessing (RSI, SMA, fill NaN, etc.)
│   └── training.py        # Model training script
├── models/
│   ├── SBER_model.pth     # Saved PyTorch model weights
│   ├── SBER_scaler_X.pkl  # MinMaxScaler for input features
│   └── SBER_scaler_y.pkl  # MinMaxScaler for target
├── notebooks/
│   ├── Best_SBER.ipynb        # Jupyter notebook with best SBER model exploration
│   └── TestBuild_tickers.ipynb# Additional experiments / analysis
└── README.md               # Project documentation (this file)
```
---

## Key Components

1. **`app/training.py`:**  
   - Trains the LSTM+Attention model on historical data  
   - Scales features and target using `MinMaxScaler`  
   - Saves the trained model (`.pth`) and scalers (`.pkl`)  

2. **`app/main.py`:**  
   - Defines the FastAPI application  
   - Handles requests to `/predict`  
   - Merges data from MOEX & CBR, applies the same preprocessing pipeline  

3. **`app/model_manager.py`:**  
   - Loads the saved model and scalers into memory  
   - Used by the API to perform inference  

4. **`models/SBER_model.pth`:**  
   - The trained PyTorch model weights for the SBER ticker  

---

## Installation

### Prerequisites

- **Python 3.8+**  
- **PyTorch** (e.g., version 1.10+)  
- **FastAPI** (e.g., version 0.95+)  
- **uvicorn** (for running the server)  
- Additional dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, etc.

### Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your_username/MOEX_PREDICT.git
   cd MOEX_PREDICT
   ```

2. **Create and Activate a Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Model Training

Train the model and generate the `.pth` and `.pkl` files:

```bash
python app/training.py
```

- **Outputs:**  
  - `SBER_model.pth`  
  - `SBER_scaler_X.pkl`  
  - `SBER_scaler_y.pkl`  

These artifacts are essential for the FastAPI server to perform predictions.

---

## Running the FastAPI App

Once the model is trained and saved, you can launch the FastAPI server:

```bash
uvicorn app.main:app --reload
```

- **Default URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)  
- **Interactive Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Usage

### Endpoint

**`POST /predict`**

### Request Payload

```json
{
  "ticker": "SBER",
  "start_date": "2025-03-27",
  "end_date": "2025-03-28"
}
```
- `ticker`: Stock symbol, e.g. `"SBER"`.
- `start_date`: Start date in `YYYY-MM-DD` format.
- `end_date`: End date in `YYYY-MM-DD` format.

### Response Example

```json
{
  "ticker": "SBER",
  "predicted_price": 230.45,
  "date": "2025-03-28"
}
```

Use the **Swagger UI** at `/docs` for an interactive way to test requests.

---

## Notebooks

- **`notebooks/Best_SBER.ipynb`:**  
  Demonstrates the best model configuration and analysis for SBER.

- **`notebooks/TestBuild_tickers.ipynb`:**  
  Contains exploratory code to fetch and preprocess multiple tickers.

These notebooks provide additional insights into data exploration, feature engineering, and model experimentation.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
