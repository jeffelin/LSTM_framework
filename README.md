
# LSTM Forecasting with Scalecast

> A powerful, modular RNN forecasting framework built with [Scalecast](https://scalecast.readthedocs.io/en/latest/).  
> Supports univariate, multivariate, probabilistic, dynamic, and transfer learning-based time series forecasting.
> Note: I am actively working on applying to new use cases and debug issues with code. src is main folder which will be contributed into. 
---

## Overview

This repository provides a generalized **LSTM-based RNN template** for forecasting future data points using time series modeling principles. Built on top of the excellent [`scalecast`](https://scalecast.readthedocs.io/en/latest/) library, this project offers a flexible forecasting pipeline with clean code, illustrative examples, and an easily extensible architecture for:

- **Univariate forecasting**  
- **Multivariate forecasting (with exogenous variables)**  
- **Probabilistic forecasting with confidence intervals**  
- **Dynamic modeling with backtesting**  
- **Transfer learning across datasets**

---

## Features

### Univariate Forecasting

Forecast using a **single target variable**, ideal for capturing basic time series patterns like trend, seasonality, and cycles.

```python
f = Forecaster(y=data['target'], current_dates=data['date'])
f.set_estimator('lstm')
f.manual_forecast()
````

---

### Multivariate Forecasting

Enhance model performance by incorporating **external regressors** (e.g., weather, holidays, marketing spend).

```python
f.add_regressors(['temperature', 'day_of_week', 'ad_spend'])
f.set_estimator('lstm')
f.manual_forecast()
```

---

### Probabilistic Forecasting

Generate **confidence intervals** around forecasted values to better quantify uncertainty.

```python
f.eval_cis(level=0.95)
f.plot('lstm', ci=True)
```

---

### Dynamic Forecasting (Backtesting)

Use **rolling or expanding windows** to evaluate forecasting performance over historical splits.

```python
f.cross_validate(k=5, method='expanding')
```

---

### Transfer Learning

Reuse trained LSTM models on new datasets or similar time series patterns.

* Fine-tune an LSTM trained on one series
* Apply to a new, but related, time series

```python
f.set_estimator('lstm')
f.manual_forecast(reuse_weights=True)
```

---

## Example Use Cases

* Forecasting **website traffic**, **electricity demand**, **stock trends**, or **retail sales**
* Understanding **forecast uncertainty** in decision-making
* Serving as a **teaching tool** for time series forecasting with deep learning 

---

## Sources

* **Scalecast Docs**: [https://scalecast-examples.readthedocs.io/en/latest/rnn/rnn.html](https://scalecast-examples.readthedocs.io/en/latest/rnn/rnn.html)
* **Datasets**: [Sun Spots (Kaggle)](https://www.kaggle.com/datasets/robervalt/sunspots)

---


