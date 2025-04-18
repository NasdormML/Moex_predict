# MOEX Price Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![FastAPI 0.115](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)

**MOEX Price Prediction** is a project that forecasts Moscow Exchange stock prices (e.g., SBER) using historical data, technical indicators (RSI, SMA), and a deep learning model built with PyTorch. The project provides an end-to-end solution—from data preprocessing and model training to serving predictions via a FastAPI REST API.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture and Structure](#architecture-and-structure)
4. [Technologies and Tools](#technologies-and-tools)
5. [Installation and Setup](#installation-and-setup)
6. [Model Training and Visualization](#model-training-and-visualization)
7. [API and Demo](#api-and-demo)
8. [License](#license)

---

## Project Overview

**MOEX Price Prediction** is designed to forecast stock prices (e.g., SBER) by combining historical market data with technical indicators like RSI and SMA. The project uses a PyTorch-based model (incorporating LSTM with an Attention mechanism) to generate predictions, which are then served through a REST API built with FastAPI.

---

## Key Features

- **End-to-End Pipeline:** Covers data fetching, preprocessing, model training, and API deployment.
- **Deep Learning Model:** Utilizes a PyTorch model for time-series forecasting.
- **Real-time Predictions:** FastAPI offers a RESTful interface for instant predictions.
- **Interactive Visualization:** Includes training graphs to monitor model performance.

---

## Architecture and Structure

```plaintext
MOEX_PREDICT/
├── app/
│   ├── __init__.py
│   ├── data.py                    # Data fetching from MOEX and CBR
│   ├── main.py                    # FastAPI entry point
│   ├── model_manager.py           # Model and scaler loading utilities
│   ├── models.py                  # PyTorch model definition (LSTM + Attention)
│   ├── monitorng.py               # MLflow check on error performance
│   ├── predict.py                 # Prediction logic using the trained model
│   ├── preprocessing.py           # Data preprocessing, RSI, SMA calculation, etc.
│   ├── scheduler.py               # Background loading of the "true" closing price (beta)
│   └── transfer_learning.py       # Retraining script
├── history/                       # Save metadate & model predict
├── models/
│   ├── v1
│   │   ├── GAZP_model.pth         # Saved PyTorch model weight
│   │   ├── GAZP_scaler_X.pkl      # RobustScaler for input features
│   │   ├── GAZP_scaler_y.pkl      # RobustScaler for input features
│   │   ├── SBER_model.pth         # Saved PyTorch model weights
│   │   ├── SBER_scaler_X.pkl      # RobustScaler for input features
│   │   └── SBER_scaler_y.pkl      # RobustScaler for target
│   └── v1.1 ...                   # New folder create after retrain models
├── notebooks/
│   ├── Best_GAZP.ipynb            # Notebook for TCN model analysis and experiments
│   └── Best_SBER.ipynb            # Notebook for LSTM model analysis and experiments
└── README.md                      # Project documentation (this file)
```

---

## Technologies and Tools

- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch 2.5.1+
- **Web Framework:** FastAPI (with Uvicorn), MLflow
- **Data Analysis Libraries:** Pandas, NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git and GitHub

---

## Installation and Setup

### Step 1. Clone the Repository

```bash
git clone https://github.com/NasdormML/Moex_predict.git
cd MOEX_PREDICT
```

### Step 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 3. Run Makefile
Makefile will set dependencies and run MLflow with Fastapi.

```bash
make run
```
MLflow will be available in [http://127.0.0.1:5001](http://127.0.0.1:5001).

Access the API at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive testing.

---

## Model Training and Visualization

Below is the PyTorch model training graph, illustrating the model's convergence and performance:

*Description:* The graph displays the progression of the loss function and key metrics as the PyTorch model learns from the training data.

# SBER ticker 

![image](https://github.com/user-attachments/assets/9ba4079f-85e3-4824-83dc-78e2757a339d)

### PyTorch Model Training Results
  | Performance Metrics | Validation Accuracy |
  |---------------------|---------------------|
  | MSE (RUB^2):        | 35.699              |
  | RMSE (RUB):         | 5.975               |
  | MAE (RUB):          | 4.589               |
  | MAPE:               | 1.74%               |

# GAZP ticker

![image](https://github.com/user-attachments/assets/1bb40b4a-f863-436a-85f4-1eb5371ee195)

### TCN Model Training Results
  | Performance Metrics | Validation Accuracy |
  |---------------------|---------------------|
  | MSE (RUB^2):        | 14.513              |
  | RMSE (RUB):         | 3.810               |
  | MAE (RUB):          | 2.727               |
  | MAPE:               | 1.89%               |
  
---

## API and Demo

### Example API Request

**Endpoint:** `POST /predict`

**Sample Request:**

```json
{
  "ticker": "SBER",
  "start_date": "2025-03-01",
  "end_date": "2025-04-05"
}
```

**Sample Response:**

```json
{
  "ticker": "SBER",
  "predicted_price": 297.791015625,
  "date": "2025-04-06"
}
```

Test the API using the interactive [Swagger UI](http://127.0.0.1:8000/docs).

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
