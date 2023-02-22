import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import metrics
import re

path = os.getcwd() + "/data/Google-Playstore-Modified.parquet"
df = pd.read_parquet(path, engine='fastparquet')

x = None
y = None

if x:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=123)
