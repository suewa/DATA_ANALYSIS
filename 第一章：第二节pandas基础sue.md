**复习：**数据分析的第一步，加载数据我们已经学习完毕了。当数据展现在我们面前的时候，我们所要做的第一步就是认识他，今天我们要学习的就是**了解字段含义以及初步观察数据**。

## 1 第一章：数据载入及初步观察

### 1.4 知道你的数据叫什么
我们学习pandas的基础操作，那么上一节通过pandas加载之后的数据，其数据类型是什么呢？

**开始前导入numpy和pandas**


```python
import numpy as np
import pandas as pd
```

#### 1.4.1 任务一：pandas中有两个数据类型DateFrame和Series，通过查找简单了解他们。然后自己写一个关于这两个数据类型的小例子🌰[开放题]

原以为series是以行，dataframe是以列，但其实并不然，注意series中的[]仍在
series是一维数组，dataframe为二维，用法可以通过 panda中文文档查看
https://www.pypandas.cn/docs/getting_started/dsintro.html#series


```python
sdata = {'Ohio': [35000,3500], 'Texas':[71000,71000] , 'Oregon': [16000,16000], 'Utah': [5000,5000] }
example_1 = pd.Series(sdata)
example_1
```




    Ohio       [35000, 3500]
    Texas     [71000, 71000]
    Oregon    [16000, 16000]
    Utah        [5000, 5000]
    dtype: object




```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
example_2 = pd.DataFrame(data)
example_2


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
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nevada</td>
      <td>2003</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.4.2 任务二：根据上节课的方法载入"train.csv"文件



```python
df = pd.read_csv('train_chinese.csv')
df.head(3)#写入代码

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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
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
  </tbody>
</table>
</div>



也可以加载上一节课保存的"train_chinese.csv"文件。通过翻译版train_chinese.csv熟悉了这个数据集，然后我们对trian.csv来进行操作
#### 1.4.3 任务三：查看DataFrame数据的每列的名称


```python
df.columns#写入代码

```




    Index(['Unnamed: 0', '乘客ID', '是否幸存', '仓位等级', '姓名', '性别', '年龄', '兄弟姐妹个数',
           '父母子女个数', '船票信息', '票价', '客舱', '登船港口'],
          dtype='object')



#### 1.4.4任务四：查看"Cabin"这列的所有值[有多种方法]


```python
df['客舱'].head(3)#这是查看索引的方法
```




    0    NaN
    1    C85
    2    NaN
    Name: 客舱, dtype: object




```python
df.客舱.head(3)#这是查看列的方法
```




    0    NaN
    1    C85
    2    NaN
    Name: 客舱, dtype: object




```python
type(df['客舱'])#上面两种都是series类型
```




    pandas.core.series.Series




```python
df[['客舱']]#此为dataframe格式
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
      <th>客舱</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C123</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>887</th>
      <td>B42</td>
    </tr>
    <tr>
      <th>888</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>889</th>
      <td>C148</td>
    </tr>
    <tr>
      <th>890</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 1 columns</p>
</div>



#### 1.4.5 任务五：加载文件"test_1.csv"，然后对比"train.csv"，看看有哪些多出的列，然后将多出的列删除
经过我们的观察发现一个测试集test_1.csv有一列是多余的，我们需要将这个多余的列删去


```python
test_1 = pd.read_csv('test_1.csv')
test_1.head(3)#写入代码

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
      <th>a</th>
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
      <td>100</td>
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
      <td>100</td>
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
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
del test_1['a']#写入代码
test_1.head(3)
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
  </tbody>
</table>
</div>



【思考】还有其他的删除多余的列的方式吗？


```python
test_1 = pd.read_csv('test_1.csv')
test_1.pop('a')#写入代码
test_1.head(3)# 思考回答
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
  </tbody>
</table>
</div>



#### 1.4.6 任务六： 将['PassengerId','Name','Age','Ticket']这几个列元素隐藏，只观察其他几个列元素


```python
df.drop(['乘客ID','姓名', '年龄','票价'],axis=1).head(3)#写入代码
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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>性别</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



恢复隐藏单元，直接查看df


```python
df.head(2)
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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
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
  </tbody>
</table>
</div>



【思考】对比任务五和任务六，是不是使用了不一样的方法(函数)，如果使用一样的函数如何完成上面的不同的要求呢？

【思考回答】

如果想要完全的删除你的数据结构，使用inplace=True，因为使用inplace就将原数据覆盖了，如果不使用inplace，那么只是返回副本


```python
df.drop(['乘客ID','姓名', '年龄','票价'],axis=1,inplace=True)
```


```python
df.head(3)
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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>性别</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### 1.5 筛选的逻辑

表格数据中，最重要的一个功能就是要具有可筛选的能力，选出我所需要的信息，丢弃无用的信息。

下面我们还是用实战来学习pandas这个功能。

#### 1.5.1 任务一： 我们以"Age"为筛选条件，显示年龄在10岁以下的乘客信息。


```python
df = pd.read_csv('train_chinese.csv')
df.head(3)#写入代码
df["年龄"]<10
```




    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    886    False
    887    False
    888    False
    889    False
    890    False
    Name: 年龄, Length: 891, dtype: bool




```python
df[df["年龄"]<10].head(5)#利用上面的索引
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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>17</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Master. Eugene</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>25</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Miss. Torborg Danira</td>
      <td>female</td>
      <td>8.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>43</th>
      <td>43</td>
      <td>44</td>
      <td>1</td>
      <td>2</td>
      <td>Laroche, Miss. Simonne Marie Anne Andree</td>
      <td>female</td>
      <td>3.0</td>
      <td>1</td>
      <td>2</td>
      <td>SC/Paris 2123</td>
      <td>41.5792</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.5.2 任务二： 以"Age"为条件，将年龄在10岁以上和50岁以下的乘客信息显示出来，并将这个数据命名为midage


```python
midage=df[(df["年龄"]>10) & (df["年龄"]<50)]#写入代码
midage.head(3)
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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
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
  </tbody>
</table>
</div>



【提示】了解pandas的条件筛选方式以及如何使用交集和并集操作


```python
midage1=df[(df["年龄"]>10) | (df["年龄"]<50)]#写入代码
midage1.shape
```




    (714, 13)




```python
midage.shape
```




    (576, 13)



#### 1.5.3 任务三：将midage的数据中第100行的"Pclass"和"Sex"的数据显示出来

【提示】在抽取数据中，我们希望数据的相对顺序保持不变，用什么函数可以达到这个效果呢？
【回答】reset_index()


```python
midage=midage.reset_index(drop=True)#写入代码
midage.head(3)
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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
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
  </tbody>
</table>
</div>




```python
midage.loc[[100],['仓位等级','性别']]
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
      <th>仓位等级</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.5.4 任务四：使用loc方法将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来


```python
midage.loc[[100,105,108],['仓位等级','姓名','性别']]#写入代码

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
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>Byles, Rev. Thomas Roussel Davids</td>
      <td>male</td>
    </tr>
    <tr>
      <th>105</th>
      <td>3</td>
      <td>Cribb, Mr. John Hatfield</td>
      <td>male</td>
    </tr>
    <tr>
      <th>108</th>
      <td>3</td>
      <td>Calic, Mr. Jovo</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.5.5 任务五：使用iloc方法将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来


```python
midage.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 576 entries, 0 to 575
    Data columns (total 13 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  576 non-null    int64  
     1   乘客ID        576 non-null    int64  
     2   是否幸存        576 non-null    int64  
     3   仓位等级        576 non-null    int64  
     4   姓名          576 non-null    object 
     5   性别          576 non-null    object 
     6   年龄          576 non-null    float64
     7   兄弟姐妹个数      576 non-null    int64  
     8   父母子女个数      576 non-null    int64  
     9   船票信息        576 non-null    object 
     10  票价          576 non-null    float64
     11  客舱          138 non-null    object 
     12  登船港口        575 non-null    object 
    dtypes: float64(2), int64(6), object(5)
    memory usage: 58.6+ KB
    


```python
midage.iloc[[100,105,108],[3,4,5]]#写入代码#写入代码

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
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>Byles, Rev. Thomas Roussel Davids</td>
      <td>male</td>
    </tr>
    <tr>
      <th>105</th>
      <td>3</td>
      <td>Cribb, Mr. John Hatfield</td>
      <td>male</td>
    </tr>
    <tr>
      <th>108</th>
      <td>3</td>
      <td>Calic, Mr. Jovo</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



【思考】对比`iloc`和`loc`的异同
iloc是按照序号，loc是按照列名
