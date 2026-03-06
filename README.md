<h1 align="center">📈 MSFT Stock Price Forecasting: A Deep Learning Approach</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

## 🚀 Overview

This repository features an advanced, robust **Long Short-Term Memory (LSTM)** neural network engineered to forecast Microsoft (MSFT) stock prices. Built strictly with industry-standard practices, this project goes far beyond a simple tutorial, demonstrating a deep understanding of time-series forecasting, quantitative finance, and intelligent deep learning architectures.

### 🌟 Key Highlights for Recruiters & Hiring Managers

- **Anti-Leakage Chronological Splitting:** Avoided the common rookie mistake of random shuffle splitting (`train_test_split`). Built a strictly chronological Training/Validation/Test pipeline (70/15/15) to prevent target leakage and "look-ahead bias."
- **Modern Market Regime Focus:** Curated the dataset to exclusively utilize the last 5 years (~1,260 trading days). This actively discards irrelevant historical data (e.g., MSFT in the 90s at $30/share) to ensure the model captures the *modern* market dynamics and volatility regime.
- **Targeted Anti-Overfitting Architecture:** Designed a highly disciplined, lightweight LSTM network (`LSTM(32) → Dense(32) → Dense(1)` with exactly 6,497 parameters). This forces the network to learn genuine financial momentum patterns rather than merely memorizing structural noise.
- **Advanced Sequencing & Preprocessing:** Engineered a rigorous 60-day sliding lookback window for feature extraction and applied strictly partitioned `MinMaxScaler` transformations to prevent gradient saturation while maintaining magnitude context.

## 🧠 Model Architecture

Financial datasets are notoriously noisy. Rather than throwing a massive, parameter-heavy network at the problem (which guarantees severe overfitting and failure on unseen data), this project utilizes a streamlined and highly optimized topology:

| Layer (Type) | Output Shape | Param # |
| :--- | :--- | :--- |
| `LSTM` | (None, 32) | 4,352 |
| `Dense` | (None, 32) | 1,056 |
| `Dense` | (None, 32) | 1,056 |
| `Dense` | (None, 1) | 33 |

**Total Trainable Parameters:** 6,497

## 📊 Dataset & Pipeline

The dataset was sourced dynamically via the `yfinance` API, covering MSFT's daily trading history. 
1. **Windowing:** Transformed absolute 1D time-series arrays into sliding 60-day matrix sequences (representing roughly a fiscal quarter of trading days), making the sequential data suitable for recurrent neural network consumption.
2. **Scaling:** Prices were normalized, ensuring the internal `tanh` and `sigmoid` gates of the LSTM cells operate within their mathematically optimal, non-saturated bounds.

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/msft-stock-prediction.git
cd msft-stock-prediction
```

### 2. Install Dependencies
Make sure you have a working Python 3.8+ environment.
```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

### 3. Run the Notebook
Launch Jupyter Notebook or JupyterLab to interact with the model:
```bash
jupyter notebook msft_stock_prediction.ipynb
```

## 📬 Contact & Connect
Designed and developed by **[Your Name]**.  
I am actively seeking roles in **Data Science, Machine Learning Engineering, and Quantitative Research**. Let's connect!

- [**LinkedIn**](https://linkedin.com/in/yourprofile)
- [**Portfolio / Website**](https://yourwebsite.com)
