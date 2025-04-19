# MOEX Price Prediction

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Pandas](https://img.shields.io/badge/pandas-2.2.3-%23150458?logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/numpy-2.2.4-%23013243?logo=numpy&logoColor=white)
![Pytorch](https://img.shields.io/badge/torch-2.6.0-%23EE4C2C?logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-2.21.3-%23004750?logo=mlflow&logoColor=white)
![Apscheduler](https://img.shields.io/badge/apscheduler-3.11.0-blue)
![Requests](https://img.shields.io/badge/requests-2.32.3-%23BA1200?logo=requests&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-0.115.12-%23009ECE?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/pydantic-2.11.3-%23008BD3?logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/uvicorn-0.34.1-%232C3E50?logo=uvicorn&logoColor=white)



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

![image](https://github.com/user-attachments/assets/8d28ae7a-80c5-411c-b087-cf9ce07bf066)

### LSTM Model Training Results
  | Performance Metrics | Validation Accuracy |
  |---------------------|---------------------|
  | MSE (RUB^2):        | 19.869              |
  | RMSE (RUB):         | 4.457               |
  | MAE (RUB):          | 3.111               |
  | MAPE:               | 1.20%               |

# GAZP ticker

![image](https://github.com/user-attachments/assets/060b4922-4ba7-4bad-9bef-d9e25f047030)

### TCN Model Training Results
  | Performance Metrics | Validation Accuracy |
  |---------------------|---------------------|
  | MSE (RUB^2):        | 18.311              |
  | RMSE (RUB):         | 4.279               |
  | MAE (RUB):          | 3.035               |
  | MAPE:               | 2.08%               |
  
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
