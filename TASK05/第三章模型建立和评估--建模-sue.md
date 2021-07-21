## 第三章 模型搭建和评估--建模

经过前面的两章的知识点的学习，我可以对数数据的本身进行处理，比如数据本身的增删查补，还可以做必要的清洗工作。那么下面我们就要开始使用我们前面处理好的数据了。这一章我们要做的就是使用数据，我们做数据分析的目的也就是，运用我们的数据以及结合我的业务来得到某些我们需要知道的结果。那么分析的第一步就是建模，搭建一个预测模型或者其他模型；我们从这个模型的到结果之后，我们要分析我的模型是不是足够的可靠，那我就需要评估这个模型。今天我们学习建模，下一节我们学习评估。

我们拥有的泰坦尼克号的数据集，那么我们这次的目的就是，完成泰坦尼克号存活预测这个任务。


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
```


```python
%matplotlib inline
```


```python
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (10, 6)  # 设置输出图片大小
```

载入这些库，如果缺少某些库，请安装他们

【思考】这些库的作用是什么呢？你需要查一查


```python
Seaborn是基于matplotlib的图形可视化python包。它提供了一种高度交互式界面，便于用户能够做出各种有吸引力的统计图表。
```


```python
%matplotlib inline
```

 **载入我们提供清洗之后的数据(clear_data.csv)，大家也将原始数据载入（train.csv），说说他们有什么不同**


```python
# 读取原数据数集
train = pd.read_csv('train.csv')
train.shape
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#读取清洗过的数据集
data = pd.read_csv('clear_data.csv')
data.shape
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



清洗数据data中没有了姓名，没有了survived数据，但多了些sex_female,sex_male这类需要分析的数据




### 模型搭建

* 处理完前面的数据我们就得到建模数据，下一步是选择合适模型
* 在进行模型选择之前我们需要先知道数据集最终是进行**监督学习**还是**无监督学习**
* 模型的选择一方面是通过我们的任务来决定的。
* 除了根据我们任务来选择模型外，还可以根据数据样本量以及特征的稀疏性来决定
* 刚开始我们总是先尝试使用一个基本的模型来作为其baseline，进而再训练其他模型做对比，最终选择泛化能力或性能比较好的模型

这里我的建模，并不是从零开始，自己一个人完成完成所有代码的编译。我们这里使用一个机器学习最常用的一个库（sklearn）来完成我们的模型的搭建

**下面给出sklearn的算法选择路径，供大家参考**


```python
# sklearn模型算法选择路径图
Image('sklearn.png')
```




    
![png](output_17_0.png)
    



【思考】数据集哪些差异会导致模型在拟合数据是发生变化


```python
噪音
异常数据
数据集数量过少

```

#### 任务一：切割训练集和测试集
这里使用留出法划分数据集

* 将数据集分为自变量（年龄 性别）和因变量（标签，比如是否存活）
* 按比例切割训练集和测试集(一般测试集的比例有30%、25%、20%、15%和10%)
* 使用分层抽样
* 设置随机种子以便结果能复现

【思考】
* 划分数据集的方法有哪些？
* 为什么使用分层抽样，这样的好处有什么？


```python
* 数据集的划分有三种方法：留出法，交叉验证法和自助法

参考资料：https://blog.csdn.net/weixin_38753213/article/details/112690712
    
*分层抽样可以保持数据分布的一致性
```

#### 任务提示1
* 切割数据集是为了后续能评估模型泛化能力
* sklearn中切割数据集的方法为`train_test_split`
* 查看函数文档可以在jupyter noteboo里面使用`train_test_split?`后回车即可看到
* 分层和随机种子在参数里寻找

要从clear_data.csv和train.csv中提取train_test_split()所需的参数


```python
from sklearn.model_selection import train_test_split
```


```python
# 一般先取出X和y后再切割，有些情况会使用到未切割的，这时候X和y就可以用,x是清洗好的数据，y是我们要预测的存活数据'Survived'
X = data
y = train['Survived']
```


```python
# 对数据集进行切割
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
```


```python
# 查看数据形状
X_train.shape, X_test.shape
```




    ((668, 11), (223, 11))



【思考】
* 什么情况下切割数据集的时候不用进行随机选取

train_test_split?
*通过该语句可以查看到文档stratify分层 当stratify=y,表示测试集和训练集的分类比例是相同的
*同一批数据，只要random_state设置地一样，那么其结果也是一样的，即结果可以被复现。

https://blog.csdn.net/weixin_45281949/article/details/102767177
https://blog.csdn.net/qq_43391414/article/details/112909834



#### 任务二：模型创建
* 创建基于线性模型的分类模型（逻辑回归）
* 创建基于树的分类模型（决策树、随机森林）
* 分别使用这些模型进行训练，分别的到训练集和测试集的得分
* 查看模型的参数，并更改参数值，观察模型变化

#### 提示
* 逻辑回归不是回归模型而是分类模型，不要与`LinearRegression`混淆
* 随机森林其实是决策树集成为了降低决策树过拟合的情况
* 线性模型所在的模块为`sklearn.linear_model`
* 树模型所在的模块为`sklearn.ensemble`


```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
```


```python
# 默认参数逻辑回归模型  %可是并没有显示参数
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

    D:\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    LogisticRegression()




```python
# 查看训练集和测试集score值
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(lr.score(X_test, y_test)))

```

    Training set score: 0.80
    Testing set score: 0.79
    


```python
# 调整参数后的逻辑回归模型
lr2 = LogisticRegression(C=100)
lr2.fit(X_train, y_train)
```

    D:\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    LogisticRegression(C=100)




```python
# 查看训练集和测试集score值
print("Training set score: {:.2f}".format(lr2.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(lr2.score(X_test, y_test)))
```

    Training set score: 0.79
    Testing set score: 0.78
    


```python
# 默认参数的随机森林分类模型
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
```




    RandomForestClassifier()




```python
print("Training set score: {:.2f}".format(rfc.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(rfc.score(X_test, y_test)))
```

    Training set score: 1.00
    Testing set score: 0.81
    


```python
# 调整参数后的随机森林分类模型
rfc2 = RandomForestClassifier(n_estimators=100, max_depth=5)
rfc2.fit(X_train, y_train)
```




    RandomForestClassifier(max_depth=5)




```python
print("Training set score: {:.2f}".format(rfc2.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(rfc2.score(X_test, y_test)))
```

    Training set score: 0.85
    Testing set score: 0.82
    

【思考】
* 为什么线性模型可以进行分类任务，背后是怎么的数学关系
* 对于多分类问题，线性模型是怎么进行分类的

* 线性模型可以以超平面的形式分割不同群。所有数据在高维空间是以点的形式，而线性模型是超平面形式

* 利用多个二分类问题解决


#### 任务三：输出模型预测结果
* 输出模型预测分类标签
* 输出不同分类标签的预测概率

#### 提示3
* 一般监督模型在sklearn里面有个`predict`能输出预测标签，`predict_proba`则可以输出标签概率


```python
# 预测标签
pred = lr.predict(X_train)
```


```python
# 此时我们可以看到0和1的数组
pred[:10]
```




    array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1], dtype=int64)




```python
# 预测标签概率
pred_proba = lr.predict_proba(X_train)
pred_proba[:10]
```




    array([[0.60732079, 0.39267921],
           [0.17972171, 0.82027829],
           [0.41573298, 0.58426702],
           [0.19148427, 0.80851573],
           [0.88010883, 0.11989117],
           [0.91382615, 0.08617385],
           [0.13339592, 0.86660408],
           [0.90661842, 0.09338158],
           [0.05300439, 0.94699561],
           [0.1096036 , 0.8903964 ]])



【思考】
* 预测标签的概率对我们有什么帮助

有助于找到合适的阈值划分哪类是正类，哪类是负类
