# Reinforced Decoder: Towards Training Recurrent Neural Networks for Time Series Forecasting
Pytorch implementation of the Reinforced Decoder. Here, a novel training and decoding approach that introduces auxiliary models and leverages reinforcement learning to guide the decoder processing, is proposed for S2S models in time series forecasting. 


## Getting Started
To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:
* Python 3.8.8
* numpy == 1.20.1
* pandas == 1.2.4
* scikit_learn == 0.24.1
* torch == 2.2.1

### Dataset
All the datasets needed for Reinforced Decoder can be obtained from the following public links, which cover both simulated and real-world time series and have different time granularities and are either univariate or multivariate.
* https://github.com/Analytics-for-Forecasting/OpenForecasting/tree/main/data/synthetic/mg. 
* https://archive.ics.uci.edu/ml/datasets/sml2010.
* https://archive.ics.uci.edu/ml/datasets/PM2.5+Data+of+Five+Chinese+Cities.
* https://github.com/XinzeZhang/TimeSeriesForecasting-torch/tree/master/data/real/ili.
* https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small.

### Examples
You can easily implement Reinforced Decoder by running the provided python file. For instance, to training the LSTM-based S2S model by Reinforced Decoder, execute the following command:
```
python run_model.py
```

## Acknowledgement
This work was supported by the National Natural Science Foundation of China (72242104),  the China Postdoctoral Science Foundation (2024M761027), and the Interdisciplinary Research Program of Hust (2024JCYJ020).

This library is constructed based on the following repos:
* https://github.com/Analytics-for-Forecasting/msvr
* https://github.com/Analytics-for-Forecasting/OpenForecasting

The S2S model with Informer and ESLSTM structures is based on the following repos: 
* https://github.com/zhouhaoyi/Informer2020
* https://github.com/zwd2016/HSN-LSTM

## Contact
For any questions, you are welcome to contact corresponding author, Yukun Bao (yukunbao@hust.edu.cn).