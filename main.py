import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functions import *
df = pd.read_csv('USA_Housing.csv')
df_X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
df_Y = df['Price']
X = np.ones((np.size(df_X, 0), np.size(df_X, 1) + 1))
X[:, 1:] = np.array(df_X.values, dtype=np.float64)
y = np.array(df_Y.values, 'float')
Theta = GradientDescent(X[:3000, :], y[:3000])