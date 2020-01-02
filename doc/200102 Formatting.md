# Formatting

파이썬 포매팅에는 `%`, ` format` 두 가지 방법이 있다.

## % 포맷팅

% 연산자와 포맷 스트링을 사용

포맷 스트링으로는 정수 `%d`, 문자열 `%s`, 실수 `%f` 가 있다.

```bash
>>> print("integer : %d, string : %s, float : %f" % (100, "str", 1.1))
integer : 100, string : str, float : 1.100000

>>> print("hi my name is %s." % "ksg")
hi my name is ksg.
```

% 연산자와 포맷 알파벳 **사이**에 **숫자**를 넣을 수도 있습니다. 문자 자릿수

```bash
# 숫자만큼 공간 확보
>>> print("my name is ~%5s~" % "ksg") 
my name is ~  ksg~
>>> print("my name is ~%5s~" % "hong")
my name is ~ hong~
>>> print("my name is ~%5s~" % "seeya")
my name is ~seeya~
```

만약 앞에다가 0을 넣으면, 공간의 빈 부분을 0으로 채워줍니다.

```bash
>>> print("number %05d" % 5) 
number 00005
>>> print("number %05d" % 400)
number 00400
>>> print("number %06d" % 9000)
number 009000
```



## format 포맷팅

`{}` 괄호를 이용한 포맷팅 방법이다.

%와 동일한 기능을 지원하며, 변수의 타입과 상관없이 **괄호**와 **숫자**만 이용하면 된다.

```bash
>>> print("integer : {}, string : {}, float : {}".format(100, "str", 1.1))
integer : 100, string : str, float : 1.100000

>>> print("integer : {0}, string : {1}, float : {2}".format(100, "str", 1.1))
integer : 100, string : str, float : 1.100000

>>> print("integer : {2}, string : {1}, float : {0}".format(1.1, "str", 100))
integer : 100, string : str, float : 1.100000
```

공간 확보 및 0으로 채우는 기능도 당연히 지원합니다.
**콜론(:)**을 기준으로 우측에 **>** 혹은 **<** 부등호를 사용해서 방향을 지정해줍니다.

```bash
>>> print("number '{0:>5d}'".format(300)) # >5d - 왼쪽(>) 다섯 공간 확보
number '  300'
>>> print("number '{0:<5d}'".format(300)) # <5d - 오른쪽(<) 다섯 공간 확보
number '300  '
>>> print("number '{0:>05d}'".format(300))
number '00300'
>>> print("number '{0:<05d}'".format(300))
number '30000'
```



## 장단점 비교

보통 문자열 포맷팅에 성능은 **"% 포맷팅"** > **"format** **포맷팅**"입니다.
% : 0.115 
format : 0.204
*(포맷팅보다 "+ 연산자"를 이용한 문자열 조합이 가장 성능적으로는 우수합니다. + : 0.067)*

% 포맷팅은 지정된 포맷 스트링으로 들어오는 값의 타입을 명시되어 있지만,
format 포맷팅은 전달되는 값의 타입을 **추측**하는 하나의 과정이 더 **추가**돼서 약간 더 느립니다.

빠른 속도, 전달되는 인자의 타입을 정확히 알고 있다면 % 포맷팅을 이용하면 좋고,
포맷 스트링을 사용하기 귀찮고 가독성이 좋은 코드를 개발하고 싶으면 format 함수를 이용하면 좋습니다.