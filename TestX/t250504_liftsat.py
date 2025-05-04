
import sys
assert sys.version_info >= (3, 8), "This script requires Python 3.8 or higher"

from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1"), "This script requires scikit-learn 0.24 or higher"

import matplotlib.pyplot as plt
plt.rc('font', size=12)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 线性模型
from sklearn.linear_model import LinearRegression

# 或者，选择k近邻模型
from sklearn.neighbors import KNeighborsRegressor

# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")  # data with DataFrame format

# 共 27 个国家，27 组数据
#print(lifesat)  # Display the first few rows of the DataFrame

X = lifesat[["GDP per capita (USD)"]].values; # 输出 numpy 数组格式
y = lifesat[["Life satisfaction"]].values

# Visualize the data
# 绘图的用法，DataFrame.plot(kind='..', x="..", y="..")
# 其中，"x" 和 "y" 是 DataFrame 的列名
# kind = "scatter" 表示散点图，kind = "line" 表示折线图
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])  # 23_500 = 23500, for easier reading
#plt.show()

# Select a linear model
# 模型选择包括制定模型，以及模型的结构
# 这里选择回归模型，再选择线性回归模型，递进关系
#model = LinearRegression()

# 选择k近邻模型，k=3
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new)) # outputs [[6.30165767]]
