# Universal Forecaster

**A modular Python and Streamlit application for time-series data collection, preprocessing, forecasting and comparative evaluation of statistical and deep learning algorithms.**

Universal Forecaster provides an interactive environment for conducting univariate and multivariate time-series forecasting experiments. The application integrates established statistical and deep learning techniques with a Direct Forecasting Strategy (DFS) developed from my earlier research in resilient cyber-physical demand forecasting.

The application supports historical data collection, preprocessing, forecasting, uncertainty estimation, visualisation and evaluation of forecasting performance.

## Key Features

### 1. Data Collection

The application supports:

- Retrieval of historical financial time-series data using Yahoo Finance through the `yfinance` Python library.
- Importing user-provided datasets in Excel or CSV format.
- Preparing datasets for subsequent preprocessing and forecasting workflows.

### 2. Data Preprocessing

The preprocessing module provides:

- Identification and validation of temporal and numerical variables.
- Detection of time-series frequency.
- Identification and treatment of missing observations.
- Construction of regular time grids.
- Comparison of interpolation methods for missing-value treatment.
- Export of processed datasets in Excel or CSV format.

### 3. Time-Series Forecasting

The application supports both **univariate forecasting**, using the historical observations of a single variable, and **multivariate forecasting**, incorporating additional explanatory or predictor variables.

Five forecasting methodologies are implemented:

| Method | Description |
|---|---|
| **LSTM** | Long Short-Term Memory neural networks for capturing temporal dependencies |
| **SARIMA-GARCH** | Statistical forecasting combining seasonal autoregressive modelling with conditional variance estimation |
| **DFS** | Wavelet decomposition and an ensemble of Convolutional Neural Networks using a direct forecasting strategy |
| **N-BEATS** | Deep neural architecture for time-series forecasting |
| **Transformer** | Attention-based neural network architecture for modelling temporal relationships |

Each methodology supports configurable forecasting parameters and experimental evaluation.

The DFS implementation draws on the forecasting methodology investigated in my published research on resilient demand forecasting for critical infrastructure.

### 4. Statistical Preprocessing and Detrending

The application implements a common statistical preprocessing framework for forecasting models, including:

- Assessment of variance stability.
- Application of logarithmic or square-root transformations where appropriate.
- Statistical assessment of differencing requirements.
- Transformation reversal to restore forecasts to their original measurement scale.
- Separate preprocessing of individual variables in multivariate datasets.

### 5. Model Evaluation and Comparison

Universal Forecaster supports experimental evaluation of forecasting algorithms using historical data.

The evaluation framework includes:

- Out-of-sample forecasting.
- Historical validation using held-out observations.
- Cross-validation procedures.
- Comparison of forecasting accuracy across different algorithms.
- Assessment of forecast interval coverage and uncertainty.
- Visual comparison of historical observations and predicted values.

The application incorporates several performance measures, including:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Mean Absolute Percentage Error (MAPE).
- Mean Arctangent Absolute Percentage Error (MAAPE).
- Directional forecasting accuracy.
- Forecast interval coverage and width.

### 6. Forecast Visualisation

The Streamlit interface provides graphical representations of:

- Historical time-series observations.
- Predicted values and forecasting horizons.
- Forecast intervals.
- Actual versus predicted values during validation.
- Statistical preprocessing and detrending choices.

These visualisations facilitate the interpretation and comparison of model behaviour.

## Software Architecture

Universal Forecaster follows a modular Python architecture.

| Module | Purpose |
|---|---|
| `app.py` | Main Streamlit application and interactive interface |
| `01_data_ingestion.py` | Historical data collection |
| `02_data_processing.py` | Data preprocessing and validation |
| `03_forecasting.py` | Shared forecasting functionality and evaluation metrics |
| `04a_forecast_plotting.py` | Forecast visualisation |
| `05_lstm_forecast.py` | LSTM forecasting |
| `06_sarima_garch.py` | SARIMA-GARCH forecasting |
| `07_dfs_forecast.py` | Direct Forecasting Strategy |
| `08_nbeats_forecast.py` | N-BEATS forecasting |
| `09_transformer_forecast.py` | Transformer forecasting |
| `10_statistical_detrending.py` | Statistical transformation and detrending |

## Technologies

**Programming language:** Python

**User interface:** Streamlit

**Data processing:** pandas, NumPy

**Statistical modelling:** SciPy, statsmodels, arch

**Machine learning and deep learning:** scikit-learn, TensorFlow

**Signal processing:** PyWavelets

**Data collection:** yfinance

**Visualisation:** Matplotlib

**Execution environments:** Local Python environment or Google Colab using Streamlit and ngrok

## Installation and Execution

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

The application can also be executed in Google Colab using an authenticated ngrok tunnel. Authentication credentials should be stored securely using Colab Secrets rather than embedded in source code.

## Research Background

The DFS component is related to my published research:

**Gheyas, I., Epiphaniou, G., Maple, C. and Lakshminarayana, S. (2022).** A Resilient Cyber-Physical Demand Forecasting System for Critical Infrastructures against Stealthy False Data Injection Attacks. *Applied Sciences*, 12(19), 10093.

https://doi.org/10.3390/app121910093

The original research investigated a resilient forecasting framework incorporating machine learning, wavelet-based processing and cybersecurity considerations.

Universal Forecaster provides a separate software implementation for exploring and comparing forecasting methodologies. It is not intended as a complete reproduction of the original cybersecurity system.

## Project Purpose

Universal Forecaster was developed as a research-oriented software platform for exploring time-series forecasting algorithms, statistical preprocessing methods, deep learning architectures and forecasting evaluation techniques.

The project demonstrates Python software development, modular application design, statistical analysis, machine learning implementation and the practical application of original forecasting research.
