# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix

import mplfinance as mplf

SEED = 1291

ohlc = pd.read_csv("archive/ohlc.csv", index_col=0, parse_dates=True)
print(ohlc.shape)
ohlc[:3]

data_df = pd.read_pickle("../input/candlestick-eda/data_df.pkl")
data_df = data_df.sort_values("imgID").reset_index(drop=True)
print(data_df.shape)
data_df[:3]

Data_Size = data_df.shape[0]
