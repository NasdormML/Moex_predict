# MOEX Price Prediction

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![FastAPI 0.95+](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)

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
│   ├── data.py            # Data fetching from MOEX and CBR
│   ├── main.py            # FastAPI entry point
│   ├── model_manager.py   # Model and scaler loading utilities
│   ├── models.py          # PyTorch model definition (LSTM + Attention)
│   ├── predict.py         # Prediction logic using the trained model
│   ├── preprocessing.py   # Data preprocessing, RSI, SMA calculation, etc.
│   └── training.py        # Model training script
├── models/
│   ├── SBER_model.pth     # Saved PyTorch model weights
│   ├── SBER_scaler_X.pkl  # MinMaxScaler for input features
│   └── SBER_scaler_y.pkl  # MinMaxScaler for target
├── notebooks/
│   └── Best_SBER.ipynb    # Notebook for model analysis and experiments
└── README.md              # Project documentation (this file)
```

---

## Technologies and Tools

- **Programming Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch
- **Web Framework:** FastAPI (with Uvicorn)
- **Data Analysis Libraries:** Pandas, NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git and GitHub

---

## Installation and Setup

### Step 1. Clone the Repository

```bash
git clone https://github.com/your_username/MOEX_PREDICT.git
cd MOEX_PREDICT
```

### Step 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4. Train the Model

Run the training script to generate the model and scaler files:

```bash
python app/training.py
```

This process creates:
- `models/SBER_model.pth`
- `models/SBER_scaler_X.pkl`
- `models/SBER_scaler_y.pkl`

### Step 5. Run the FastAPI Application

```bash
uvicorn app.main:app --reload
```

Access the API at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive testing.

---

## Model Training and Visualization

Below is the PyTorch model training graph, illustrating the model's convergence and performance:

![image](https://github.com/user-attachments/assets/9418a732-f4d2-4e9d-9522-86d33856a6f9)

*Description:* The graph displays the progression of the loss function and key metrics as the PyTorch model learns from the training data.
### PyTorch Model Training Results
  | Performance Metrics | Validation Accuracy |
  |---------------------|---------------------|
  | MSE (RUB^2):        | 48.849              |
  | RMSE (RUB):         | 6.989               |
  | MAE (RUB):          | 5.377               |
  | MAPE:               | 2.08%               |

---

## API and Demo

### Example API Request

**Endpoint:** `POST /predict`

**Sample Request:**

```json
{
  "ticker": "SBER",
  "start_date": "2025-02-27",
  "end_date": "2025-03-30"
}
```

**Sample Response:**

```json
{
  "ticker": "SBER",
  "predicted_price": 310.77154541015625,
  "date": "2025-03-30"
}
```

Test the API using the interactive [Swagger UI](http://127.0.0.1:8000/docs).

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
