import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 案例分析步骤
# 1.获取数据
# 2.基本数据处理
# 2.1 缺失值处理
# 2.2 确定特征值,目标值
# 2.3 分割数据
# 3.特征工程(标准化)
# 4.机器学习(逻辑回归)
# 5.模型评估

def dm01_LogisticRegression():
    # 1. 获取数据.
    data = pd.read_csv('../data/breast-cancer-wisconsin.csv')
    data.info()

    # 2. 数据预处理.
    # data = data.replace(to_replace='?', value=np.nan)
    data = data.replace('?', np.nan)
    data = data.dropna()
    data.info()

    # 3. 确定特征值和目标值.
    x = data.iloc[:, 1:-1]
    y = data.Class
    print(f'x.head(): {x.head()}')
    print(f'y.head(): {y.head()}')

    # 3. 分割数据.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
    # 4. 特征处理.
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 5. 模型训练.
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # 6. 模型预测
    y_predict = estimator.predict(x_test)
    print(f'预测值: {y_predict}')

    # 7. 模型评估
    print(f'准确率: {estimator.score(x_test, y_test)}')
    print(f'准确率: {accuracy_score(y_test, y_predict)}')

if __name__ == '__main__':
    dm01_LogisticRegression()
