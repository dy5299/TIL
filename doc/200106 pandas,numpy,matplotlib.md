# 1. 판다스(Pandas)

판다스(Pandas)는 파이썬 데이터 처리를 위한 라이브러리

```bash
import pandas as pd
```

Pandas는 총 세 가지의 데이터 구조를 사용합니다.

1. 시리즈(Series)
2. 데이터프레임(DataFrame)
3. 패널(Panel)

이 중 데이터프레임이 가장 많이 사용

### 1) 시리즈(Series)

인덱스(index)와 값(values)으로 구성

### 2) 데이터프레임(DataFrame)

데이터프레임은 2차원 리스트를 매개변수로 전달합니다.

열(columns), 인덱스(index), 값(values)으로 구성됩니다.

### 3) 데이터프레임의 생성

데이터프레임은 리스트(List), 시리즈(Series), 딕셔너리(dict), Numpy의 ndarrays, 또 다른 데이터프레임으로 생성할 수 있습니다.

```bash
# 리스트로 생성하기
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]
df = pd.DataFrame(data)
# column 지정
pd.DataFrame(data, columns=['학번', '이름', '점수'])
```

```bash
# 딕셔너리로 생성하기
data = { '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
'이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
         '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]}

df = pd.DataFrame(data)
```

### 4) 데이터프레임 조회하기

아래의 명령어는 데이터프레임에서 원하는 구간만 확인하기 위한 명령어로서 유용하게 사용됩니다.

df.head(n) - 앞 부분을 n개만 보기
df.tail(n) - 뒷 부분을 n개만 보기
df['열이름'] - 해당되는 열을 확인

### 5) 외부 데이터 읽기

Pandas는 CSV, 텍스트, Excel, SQL, HTML, JSON 등 다양한 데이터 파일을 읽고 데이터 프레임을 생성할 수 있습니다.

```bash
df=pd.read_csv('example.csv 파일의 경로')
```

# 2. 넘파이(Numpy)

```bash
import numpy as np
```

Numpy의 주요 모듈은 아래와 같습니다.

1. np.array() # 리스트, 튜플, 배열로 부터 ndarray를 생성

2. np.asarray() # 기존의 array로 부터 ndarray를 생성

3. np.arange() # range와 비슷

4. np.linspace(start, end, num) # [start, end] 균일한 간격으로 num개 생성

5. np.logspace(start, end, num) # [start, end] log scale 간격으로 num개 생성

### 1) np.array()

Numpy의 핵심은 ndarray

np.array()는 리스트, 튜플, 배열로 부터 ndarray를 생성합니다.

또한 인덱스가 항상 0으로 시작한다는 특징을 갖고 있습니다.

### 2) ndarray의 초기화

`zeros()` : 해당 배열에 모두 0을 삽입

`ones()` : 모두 1을 삽입

`full()` :  배열에 사용자가 지정한 값을 삽입

`eye()` : 기본행렬 (2차원)

### 3) np.arange()

지정해준 범위에 대해서 배열을 생성

numpy.arange(start, stop, step, dtype)
a = np.arange(n) # 0, ..., n-1까지 범위의 지정.
a = np.arange(i, j, k) # i부터 j-1까지 k씩 증가하는 배열.

```bash
np.arange(10) #0부터 9까지
np.arange(1, 10, 2) #1부터 9까지 +2씩 적용되는 범위
```

### 4) reshape()

일차원 배열을 다차원으로 변형

### 5) Numpy 슬라이싱

ndarray를 통해 만든 다차원 배열은 파이썬의 리스트처럼 슬라이스(Slice) 기능을 지원합니다. 슬라이스 기능을 사용하면 원소들 중 복수 개에 접근할 수 있습니다.

```bash
a[0:2, 0:2] #행,열
a[0, :] #첫번째행, 모든 열
```

### 6) Numpy 정수 인덱싱(integer indexing)

정수 인덱싱은 원본 배열로부터 부분 배열을 구합니다.

```bash
a[[2, 1],[1, 0]] # a[[row2, col1],[row1, col0]
```

### 7) Numpy 연산

Numpy를 사용하면 배열간 연산을 손쉽게 수행할 수 있습니다. +, -, *, /의 연산자를 사용할 수 있으며, 또는 add(), substract(), multiply(), divide() 함수를 사용할 수도 있습니다.

위에서 *를 통해 수행한 것은 요소별 곱이었습니다. Numpy에서 벡터와 행렬의 곱 또는 행렬곱을 위해서는 dot()을 사용해야 합니다.

```bash
np.dot(a, b)
```

# 3. 맷플롯립(Matplotlib)

데이터를 차트(chart)나 플롯(plot)으로 시각화(visulaization)하는 패키지입니다. 데이터 분석에서 Matplotlib은 데이터 분석 이전에 데이터 이해를 위한 시각화나, 데이터 분석 후에 결과를 시각화하기 위해서 사용됩니다.

```bash
%matplotlib inline
import matplotlib.pyplot as plt
```

주피터 노트북에 그림을 표시하도록 지정하는 %matplotlib inline 또한 우선 수행해야 합니다.

### 1) 라인 플롯 그리기 - plot()

```bash
plt.title('students')
plt.plot([1,2,3,4],[2,4,8,6])
plt.plot([1.5,2.5,3.5,4.5],[3,5,8,10]) #라인 새로 추가
plt.xlabel('hours') #축 레이블
plt.ylabel('score')
plt.legend(['A student', 'B student']) #범례
plt.show()
```



