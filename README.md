# Hybrid EGARCH–TCN–Attention Model for Financial Volatility Forecasting

## Overview
This project implements a **Hybrid Financial Risk Forecasting Framework** that combines **Econometric Models** and **Deep Learning Models** to improve volatility and tail-risk prediction in financial markets.

The framework integrates:

- **EGARCH-GED** for volatility modeling
- **Temporal Convolutional Network (TCN)** for temporal pattern extraction
- **Multi-Head Attention** to focus on important market events

The model predicts **Value at Risk (VaR)** and **Expected Shortfall (ES)** for financial assets.

---

# Architecture

Hybrid Pipeline:

```
Market Data (OHLCV)
        ↓
   Preprocessing
        ↓
EGARCH-GED Volatility
        ↓
   TCN Network
        ↓
 Multi-Head Attention
        ↓
    Dense Layer
        ↓
 Volatility Forecast
        ↓
   VaR & ES Risk
```

---

# Features

- Hybrid **Econometric + Deep Learning** model
- Captures **asymmetric volatility**
- Handles **heavy-tailed distributions**
- Sliding window **time series forecasting**
- Predicts:
  - Volatility
  - Value-at-Risk (VaR)
  - Expected Shortfall (ES)
- Backtested using **Christoffersen Test**

---

# Datasets

Two financial datasets were used:

| Dataset | Description | Period |
|------|------|------|
| CSI-300 | Chinese stock market index | 2015 – 2023 |
| BTC-USD | Bitcoin price dataset | 2015 – 2023 |

Features used:

```
Date
Open
High
Low
Close
Volume
```

---

# Data Preprocessing

### Log Return Calculation

```
Rt = ln(Pt) − ln(Pt−1)
```

### Feature Scaling

```
Min-Max Scaling
```

### Sliding Window

```
Window size = 10 days
```

### Statistical Diagnostics

- ADF Test (Stationarity)
- Jarque-Bera Test (Heavy tails)
- ARCH-LM Test (Heteroskedasticity)

---

# Benchmark Models

The hybrid model is compared with:

- VAR
- LSTM
- TCN

---

# Model Performance

### BTC-USD Results

| Model | RMSE | R² |
|------|------|------|
| VAR | 0.2348 | 0.0057 |
| LSTM | 0.0462 | 0.0764 |
| TCN | 0.0461 | 0.0721 |
| Hybrid Model | **0.0386** | **0.3407** |

### CSI-300 Results

| Model | RMSE | R² |
|------|------|------|
| VAR | 0.1325 | 0.0013 |
| LSTM | 0.0547 | 0.0381 |
| TCN | 0.0555 | 0.0721 |
| Hybrid Model | **0.0527** | **0.2999** |

---

# Risk Metrics

### Value at Risk (VaR)

```
VaR(α) = μt + σt qν(α)
```

### Expected Shortfall (ES)

```
ES = μt − σt fν(qν(α)) / α
```

Confidence Level used:

```
95%
```

---

# Christoffersen Backtesting

| Test | p-value |
|-----|-----|
| Unconditional Coverage | 0.919 |
| Independence | 0.938 |
| Conditional Coverage | 0.992 |

Results show that the model produces **statistically reliable VaR predictions**.

---

# Project Structure

```
project/
│
├── data/
│
├── preprocessing/
│
├── models/
│   ├── EGARCH.py
│   ├── TCN.py
│   └── HybridModel.py
│
├── training/
│
├── evaluation/
│
└── main.py
```

---

# Technologies Used

- Python
- NumPy
- TensorFlow / PyTorch
- Scikit-learn
- Pandas
- Statsmodels

---

# Applications

This framework can be used for:

- Portfolio Risk Management
- Algorithmic Trading
- Financial Forecasting
- Crypto Market Analysis
- Regulatory Risk Assessment

 
