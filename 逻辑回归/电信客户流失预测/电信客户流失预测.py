import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# 1. 定义函数, 表示: 数据基本处理
def dm01_数据基本处理():
    # 1. 读取数据, 查看数据的基本信息.
    churn_pd = pd.read_csv('data/churn.csv')
    # churn_pd.info()
    # print(f'churn_pd.describe(): {churn_pd.describe()}')
    # print(f'churn_pd: {churn_pd}')

    # 2. 处理类别型的数据, 类别型数据做 one-hot编码(热编码).
    churn_pd = pd.get_dummies(churn_pd)
    churn_pd.info()
    # print(f'churn_pd: {churn_pd}')

    # 3. 去除列 Churn_No, gender_Male
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)  # 按列删除
    print(f'churn_pd: {churn_pd}')

    # 4. 列标签重命名, 打印列名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    print(f'列名: {churn_pd.columns}')

    # 5. 查看标签的分布情况 0.26用户流失
    value_counts = churn_pd.flag.value_counts()
    print(value_counts)


# 2. 定义函数, 表示: 特征筛选
def dm02_特征筛选():
    # 1. 读取数据
    churn_pd = pd.read_csv('data/churn.csv')
    # 2. 处理类别型的数据, 类别型数据做 one-hot编码(热编码).
    churn_pd = pd.get_dummies(churn_pd)
    # 3. 去除列 Churn_No, gender_Male
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 4. 列标签重命名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 5. 查看标签的分布情况
    value_counts = churn_pd.flag.value_counts()
    print(value_counts)
    # 6. 查看Contract_Month 是否预签约流失情况
    sns.countplot(data=churn_pd, x='Contract_Month', hue='flag')
    plt.show()


# 3. 定义函数, 表示: 模型训练 和 评测
def dm03_模型训练和评测():
    # 1. 读取数据
    churn_pd = pd.read_csv('data/churn.csv')

    # 2. 数据预处理
    # 2.1 处理类别型的数据, 类别型数据做 one-hot编码(热编码).
    churn_pd = pd.get_dummies(churn_pd)
    # 2.2 去除列 Churn_No, gender_Male
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 2.3 列标签重命名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    # 3. 特征处理.
    # 3.1 提取特征和标签
    x = churn_pd[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_pd['flag']
    # 3.2 训练集和测试集的分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)

    # 4. 模型训练.
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 5. 模型预测
    y_predict = estimator.predict(x_test)
    print(f'预测结果: {y_predict}')

    # 6. 模型评估
    print(f'准确率: {accuracy_score(y_test, y_predict)}')
    print(f'准确率: {estimator.score(x_test, y_test)}')

    # 计算AUC值.
    print(f'AUC值: {roc_auc_score(y_test, y_predict)}')



if __name__ == '__main__':
    # dm01_数据基本处理()
    # dm02_特征筛选()
    dm03_模型训练和评测()
