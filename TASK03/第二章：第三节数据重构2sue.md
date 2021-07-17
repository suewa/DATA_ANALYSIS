**复习：**在前面我们已经学习了Pandas基础，第二章我们开始进入数据分析的业务部分，在第二章第一节的内容中，我们学习了**数据的清洗**，这一部分十分重要，只有数据变得相对干净，我们之后对数据的分析才可以更有力。而这一节，我们要做的是数据重构，数据重构依旧属于数据理解（准备）的范围。

#### 开始之前，导入numpy、pandas包和数据


```python
# 导入基本库
import numpy as np
import pandas as pd
```


```python
# 载入上一个任务人保存的文件中:result.csv，并查看这个文件
text = pd.read_csv('result.csv')
text.head()
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
      <th>Unnamed: 0</th>
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
      <td>0</td>
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
      <td>1</td>
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
      <td>2</td>
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
      <td>3</td>
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
      <td>4</td>
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



# 2 第二章：数据重构


## 第一部分：数据聚合与运算

### 2.6 数据运用

#### 2.6.1 任务一：通过教材《Python for Data Analysis》P303、Google or anything来学习了解GroupBy机制

Groupby
（1）根据用户提供的一个或多个键，将数据(dataframe,series等类型)分离到各个组中
（2）将函数就应用到各个组中，产生新的值
https://www.shulanxt.com/analytics/python/data-groupby

在了解GroupBy机制之后，运用这个机制完成一系列的操作，来达到我们的目的。

下面通过几个任务来熟悉GroupBy机制。

#### 2.4.2：任务二：计算泰坦尼克号男性与女性的平均票价


```python
df  = text['Fare'].groupby(text['Sex'])
means = df.mean()
means

```




    Sex
    female    44.479818
    male      25.523893
    Name: Fare, dtype: float64



grouped= text['data'].groupby(text['key1'],text['key2'])
（1）分组：根据key1,key2对列data进行分组
（2）利用函数对每个分组计算结果（如mean sum）

#### 2.4.3：任务三：统计泰坦尼克号中男女的存活人数


```python
survived_sex = text['Survived'].groupby(text['Sex']).sum()
survived_sex
```




    Sex
    female    233
    male      109
    Name: Survived, dtype: int64



#### 2.4.4：任务四：计算客舱不同等级的存活人数


```python
survived_pclass = text['Survived'].groupby(text['Pclass']).sum()
survived_pclass
```

【**提示：**】表中的存活那一栏，可以发现如果还活着记为1，死亡记为0,因此可以使用sum函数

【**思考**】从数据分析的角度，上面的统计结果可以得出那些结论


```python
女性一般会更愿意花费更昂贵的票价，女性的存活人数是男性两倍，二等仓存活人数最少

```

【思考】从任务二到任务三中，这些运算可以通过agg()函数来同时计算。并且可以使用rename函数修改列名。你可以按照提示写出这个过程吗？


```python
text.groupby('Sex').agg({'Fare': 'mean', 'Pclass': 'count'}).rename(columns=
                                 {'Fare': 'mean_fare', 'Pclass': 'count_pclass'})



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
      <th>mean_fare</th>
      <th>count_pclass</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>44.479818</td>
      <td>314</td>
    </tr>
    <tr>
      <th>male</th>
      <td>25.523893</td>
      <td>577</td>
    </tr>
  </tbody>
</table>
</div>



agg用法：text.groupby('key').agg({'data1': 'f1()', 'data2': 'f2()'})


#### 2.4.5：任务五：统计在不同等级的票中的不同年龄的船票花费的平均值


```python
text.groupby(['Pclass','Age'])['Fare'].mean().head()
```




    Pclass  Age  
    1       0.92     151.5500
            2.00     151.5500
            4.00      81.8583
            11.00    120.0000
            14.00    120.0000
    Name: Fare, dtype: float64




```python
#df  = text['Fare'].groupby(text['Pclass'],text['Age'])
#means = df.mean()

```

#方法二，利用grouped= text['data'].groupby(text['key1'],text['key2'])的语法，可是却显示错误TypeError: 'Series' objects are mutable, thus they cannot be hashed

#### 2.4.6：任务六：将任务二和任务三的数据合并，并保存到sex_fare_survived.csv


```python
result = pd.merge(means,survived_sex,on='Sex')
result
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
      <th>Fare</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>44.479818</td>
      <td>233</td>
    </tr>
    <tr>
      <th>male</th>
      <td>25.523893</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>



利用sex索引作为共同列可以拼接，on='Sex'

result =means.join(survived_sex)是错误的，因为means为series，无法适用join（其用于dataframe）

#### 2.4.7：任务七：得出不同年龄的总的存活人数，然后找出存活人数的最高的年龄，最后计算存活人数最高的存活率（存活人数/总人数）



```python
#不同年龄的存活人数
survived_age = text['Survived'].groupby(text['Age']).sum()
survived_age.head(10)

```




    Age
    0.42    1
    0.67    1
    0.75    2
    0.83    2
    0.92    1
    1.00    5
    2.00    3
    3.00    5
    4.00    7
    5.00    4
    Name: Survived, dtype: int64




```python
#找出最大值的年龄段
survived_age[survived_age.values==survived_age.max()]

```




    Age
    24.0    15
    Name: Survived, dtype: int64




```python
_sum = text['Survived'].sum()
print("sum of person:"+str(_sum))
precent =survived_age.max()/_sum

print("最大存活率："+str(precent))
```

    sum of person:342
    最大存活率：0.043859649122807015
    
