# PoliMi-Time-Series-Forecasting

Welcome to the **PoliMi Time Series Forecasting Challenge** repository! This project is part of the Artificial Neural Networks and Deep Learning course at **Politecnico di Milano**. The objective is to develop models capable of predicting future values in uncorrelated time series, with a focus on generalization across various time domains.

## Project Description

In this challenge, students are tasked with:
- Building forecasting models to predict future samples of input time series.
- Ensuring the model’s generalization ability across different temporal contexts.

### Dataset Details

The dataset consists of monovariate time series (single feature) from six different domains, structured as follows:

- **Time Series Length**: Each series is padded to a maximum length of 2776 for uniformity, though individual lengths vary.
- **Data Format**: `.npy` files located in the `data/` folder.
- **Categories**: Six distinct categories labeled 'A', 'B', 'C', 'D', 'E', and 'F'.

#### Dataset Structure

The dataset is contained in the `data/` folder with the following files:
- `data/training_data.npy`: Numpy array of shape (48000, 2776), containing 48000 time series.
- `data/valid_periods.npy`: Numpy array of shape (48000, 2), indicating the start and end indices for each time series (non-padded portions).
- `data/categories.npy`: Numpy array of shape (48000,), containing the category code for each time series.

**Note**: Each time series category is unique to its data source, with no explicit relation among series across domains.

### Data Loading

Load the dataset files directly from the `data/` folder:
```python
import numpy as np
training_data = np.load("data/training_data.npy")
valid_periods = np.load("data/valid_periods.npy")
categories = np.load("data/categories.npy")

