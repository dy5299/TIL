# Database

## intro

Django DB engine을 사용할 수밖에 없다.

DB는 거의 오라클 95% 이상

10년, 20년이 지나도 DB는 변화가 크게 없다.

관계형 DB로 거의 모든걸 다 할 수 있기 때문에.

객체형 DB를 오라클이 내놓아도 사람들이 잘 안 씀

관계형 DB가 너무 오랫동안 지배해와서 더 효율적인 계층형 DB가 사용될 기회가 없었다고 본다.



- 데이터-정보-지식-지혜

데이터는 사실들 그 자체에 대한 일차적인 표현이나, 정보란 사실들과 이들로부터 유도될 수 있는 사실들을 의미.

지식은 data와 info보다 더 상위 수준의 개념인데, 이들을 처리하는 바업이나 어떤 근거에 의한 판단을 내리는데 필요한 분석과 판단에 관한 법칙 등을 포함한다.

데이터(사실,관찰. noting)

-> 정보(상황설명을 위한 데이터.판단.+통계. value)

-> 지식(의미있는정보.의도포함.+전략,방법론. money)

-> 지혜(통찰력을 가진 지식. +철학,문화. bob)



- database는 지속적인 데이터에 속함.



- 데이터베이스 관리 시스템
  - DBMS, databse management system : 컴퓨터에 저장되는 데이터베이스를 고나리해주는 소프트웨어 시스템
  - 상용 DBMS: DB2, Oracle, Informix, BADA, MS SQL Server, Sybase, dBase, FoxPro, Access, ...

- 창고관리인이 DBMS이다.

SQLite: 안드로이드, iOS에 사용. 프리 라이센스라서 영속성 데이터베이스가 필요한 거의 모든 곳에서 사용된다고 보면 된다.



- 데이터베이스의 특징

  - 데이터의 무결성

  - 데이터의 독립성

  - 보안

  - 데이터 중복 최소화

  - 응용 프로그램 제작 및 수정 용이: 통일된 방식으로 응용 프로그램을 작성 가능, 유지보수 또한 쉬움

  - 데이터의 안전성 향상: 손상되어도 복원 및 복구 가능



- 관계형 DBMS
  - 모든 데이터는 **테이블**에 저장
  - **테이블 간의 관계**는 기본키(PK)와 외래키(FK)를 사용하여 맺음 (부모-자식 관계)
  - 다른 DBMS에 비해 업무 변화에 따라 바로 순응할 수 있고 유지보수 측면에서도 편리
  - 데용량 데이터를 체계적으로 관리할 수 있음
  - 데이터의 무결성도 잘 보장됨
  - 시스템 자원을 많이 차지하여 시스템이 전반적으로 느려지는 단점이 있음.
- 계층형 DBMS  (컴퓨터 대부분, 초창기 많이 사용됨. ex. dictionary)
  - 각 계층이 **트리** 형태를 띠고 1:N 관계를 가짐
  - 한번 구축하면 구조를 변경하기 까다로움
  - 접근의 유연성이 부족하여 임의 검색 시 어려움



데이터의 중복이 발생하면 절대 안 된다

## practice

- 중복 데이터가 많아

| 고객이름 | 날짜     | 총합  | 품목 | 단가 | 수량 | 합   |
| -------- | -------- | ----- | ---- | ---- | ---- | ---- |
| 홍길동   | 2020.2.7 | 10000 | 사과 | 1000 | 5    | 5000 |
| 홍길동   | 2020.2.7 | 10000 | 배   | 2000 | 2    | 4000 |

- 중복 제거 위해 but 품목 개수가 정해져 있을 때만 가능하지 않을까

| 고객이름 | 날짜     | 총합  | 품목 | 단가 | 수량 | 합   | 품목2 | 단가2 | 수량2 | 합2  |
| -------- | -------- | ----- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ---- |
| 홍길동   | 2020.2.7 | 10000 | 사과 | 1000 | 5    | 5000 | 배    | 2000  | 2     | 4000 |

테이블 데이터베이스의 한계.

- 관계형 데이터베이스: 테이블은 테이블로 정의하고, 관계를 통해서 DB를 설정.

테이블을 여러 개로 나누고 관계를 맺어준다(공통의 고유 key를 부여)

테이블1

| 고유번호 | 고객이름 | 날짜      | 총합  |
| -------- | -------- | --------- | ----- |
| 1        | 홍길동   | 2020.2.7  | 10000 |
| 2        | 이순신   | 2020.2.15 | 20000 |

테이블2

| 고유번호 | 품목 | 단가 | 수량 | 합   |
| -------- | ---- | ---- | ---- | ---- |
| 1        | 사과 | 1000 | 5    | 5000 |
| 1        | 배   | 2000 | 2    | 4000 |
| 2        | 연필 | 100  | 4    | 400  |

두 테이블을 join(곱하기)하여 하나의 테이블로 만들 수 있다.



dictionary로 만든다면...

```python
dict = [  {'name':홍길동, 'date':'2020.2.7',
          'product':[{'품목':'사과', 'price':1000, '수량':5}, 
                    {'품목':'배', 'price':2000, '수량':2}]},
          {'name':이순신, 'date':'2020.2.15',
          'product':[{'품목':'연필', 'price':100, '수량':4}]}
       ]
```

관계설정 할 필요가 없다. 중복 문제도 없다.



- SQL: 데이터베이스를 조작하는 언어

- SQL의 특징
  - DBMS 제작 회사와 독립적임
  - 다른 시스템으로의 이식성이 좋음
  - 표준이 계속 발전함

  - 대화식 언어임
  - 클라이언트/서버 구조 지원함

- SQL과 파이썬의 차이:

언어는 크게 두 가지 형태로 나뉘어진다. 절차식 언어와 선언적 언어

지금까지 배웠던 모든 언어는 절차식 언어다. how to를 명시하는 것이 절차식 언어이다.

선언적 언어는 원하는 목적만 말하는 것이다.

SQL은 선언적 언어이다. 알고리즘을 붙여 설명하지 않고, 찾을 것을 말하면 어떻게 할 지는 DB운영자가 알아서 가장 조건에 맞는 데이터를 리턴해준다.

- SQL 구문 형식

SELECT 열이름 FROM 테이블이름 WHERE 조건절

```sql
SELECT userID, userName FROM userTBL WHERE birthYear >= 2000;
```

## Django 에서 DB 생성

```bash
python manage.py createsuper
```

Django administration	http://localhost:8000/admin/

SQLite browser : 관리 툴.



## SQL 문법

- 테이블 생성

```sql
CREATE TABLE userTBL -- 회원 테이블 
( userID CHAR(8) NOT NULL PRIMARY KEY, -- 사용자 아이디(PK) 
  userName VARCHAR(10) NOT NULL, -- 이름 
  birthYear INT NOT NULL, -- 출생 연도 
  addr CHAR(2) NOT NULL, -- 지역(경기, 서울, 경남 식으로 2글자만 입력) 
  mobile1 CHAR(3), -- 휴대폰의 국번(011, 016, 017, 018, 019, 010 등) 
  mobile2 CHAR(8), -- 휴대폰의 나머지 번호(하이픈 제외) 
  height SMALLINT, -- 키 
  mDate DATE -- 회원 가입일
);

CREATE TABLE buyTBL -- 구매 테이블 
( num INTEGER   PRIMARY KEY AUTOINCREMENT, 
  userID CHAR(8) NOT NULL, -- 아이디(FK) 
  prodName CHAR(6) NOT NULL, -- 물품 
  groupName CHAR(4), -- 분류 
  price INT NOT NULL, -- 단가 
  amount SMALLINT NOT NULL, -- 수량 
  FOREIGN KEY (userID) REFERENCES userTBL (userID)
);
```

CHAR(8) : 자릿수 8자리 고정

NOT NULL: 값 필수

PRIMARY KEY: 중복값 허용 안 함



AUTO_INCREMENT: 알아서 값이 증가

FOREIGN KEY: 두 데이터베이스의 관련을 맺어주는 키를 정의



- ROW 등록

```sql
INSERT INTO userTBL VALUES
('KHD', '강호동', 1970, '경북', '011', '22222222', 182, '2017-7-7');
```

column 순서대로 입력해야 한다.

NOT NULL, PRIMARY KEY: 설계된 원칙대로 database가 저장된다는 것을 보장해준다.

```	sql
INSERT INTO buyTBL VALUES
(NULL, 'fef', '운동화', NULL, 30, 2);
```

오류난다. NOT NULL 문제이기도 하지만, 'fef'라는 유저가 없어서. 

```sql
INSERT INTO buyTBL (num,userID,prodName,price,amount) VALUES
(1, 'KHD', '운동화', 30, 2);
INSERT INTO buyTBL (num, userID,prodName,price,amount) VALUES
(2, 'KHD', '노트북', 1000, 1);
```

일부만 명시할 경우 맵핑을 직접 해줘야한다.



- group by

```sql
SELECT addr, count(*) FROM userTBL GROUP BY addr
```



- 데이터 등록

```sql
INSERT INTO userTBL VALUES ('KHD', '강호동', 1970, '경북', '011', '2222', 182, '2007-7-7');
INSERT INTO userTBL VALUES ('KKJ', '김국진', 1965, '서울', '019', '33333333', 171, '2009-9-9');
INSERT INTO userTBL VALUES ('KYM', '김용만', 1967, '서울', '010', '44444444', 177, '2015-5-5');
INSERT INTO userTBL VALUES ('KJD', '김제동', 1974, '경남', NULL , NULL, 173, '2013-3-3');
INSERT INTO userTBL VALUES ('NHS', '남희석', 1971, '충남', '016', '66666666', 180, '2017-4-4');
INSERT INTO userTBL VALUES ('SDY', '신동엽', 1971, '경기', NULL, NULL, 176, '2008-10-10');
INSERT INTO userTBL VALUES ('LHJ', '이휘재', 1972, '경기', '011', '88888888', 180, '2006-4-4');
INSERT INTO userTBL VALUES ('LKK', '이경규', 1960, '경남', '018', '99999999', 170, '2004-12-12');
INSERT INTO userTBL VALUES ('PSH', '박수홍', 1970, '서울', '010', '00000000', 183, '2012-5-5');

INSERT INTO buyTBL VALUES (NULL, 'KHD', '운동화', NULL, 30, 2);
INSERT INTO buyTBL VALUES (NULL, 'KHD', '노트북', '전자', 1000, 1);
INSERT INTO buyTBL VALUES (NULL, 'KYM', '모니터', '전자', 200, 1);
INSERT INTO buyTBL VALUES (NULL, 'PSH', '모니터', '전자', 200, 5);
INSERT INTO buyTBL VALUES (NULL, 'KHD', '청바지', '의류', 50, 3);
INSERT INTO buyTBL VALUES (NULL, 'PSH', '메모리', '전자', 80, 10);
INSERT INTO buyTBL VALUES (NULL, 'KJD', '책', '서적', 15, 5);
INSERT INTO buyTBL VALUES (NULL, 'LHJ', '책', '서적', 15, 2);
INSERT INTO buyTBL VALUES (NULL, 'LHJ', '청바지', '의류', 50, 1);
INSERT INTO buyTBL VALUES (NULL, 'PSH', '운동화', NULL, 30, 2);
INSERT INTO buyTBL VALUES (NULL, 'LHJ', '책', '서적', 15, 1);
INSERT INTO buyTBL VALUES (NULL, 'PSH', '운동화', NULL, 30, 2);
```

(auto increment 작동이 안 되어서 직접 값을 넣음)



```sql
SELECT userID, sum(price*amount) FROM buyTBL GROUP BY userID
```



- join

2개 이상의 테이블을 묶어서 하나의 결과 테이블을 만드는 것.

innerjoin

```sql
SELECT userTBL.userID, username, sum(amount), sum(amount*price) as total
FROM userTBL, buyTBL 
WHERE userTBL.userID = buyTBL.userID
GROUP BY userTBL.userID, username
ORDER BY total
```

userID는 두 테이블 모두에 있으므로 기준으로 할 테이블을 지정해줘야 오류가 안 난다.

```python
SELECT userTBL.userID, username, sum(amount), sum(amount*price) as total
FROM userTBL as u, buyTBL as b
WHERE u.userID = b.userID
GROUP BY addr
having total>170
ORDER BY total

```





cf

```sql
INSERT INTO userTBL VALUES
('YJS', '유재석', 1972, '서울', '010' , '1111111', 178, '2008-8-8');
```



- ORM

Object Relation Model

~



## Django에 DB 사용하기

```bash
#관리자 및 user 생성
python manage.py createsuperuser
#mysite/settings.py 설정
INSTALLED_APPS = [, ..., myapp]				#앱 추가
```



DB 관련 설정은 app별로 설정

```python
#model 클래스 생성 myapp/models.py
from django.db import models


class User(models.Model) :
    userid = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=10)
    age = models.IntegerField()
    hobby = models.CharField(max_length=20)

    def __str__(self):		#print 적용할 때 자동으로 적용되는 함수
        return f"{self.userid} / {self.name} / {self.age}"


#admin.py 수정
from django.contrib import admin
from myapp.models import User
admin.site.register(User)
```

```python
#myapp/apps.py 자동으로 생성되어있어
from django.apps import AppConfig

class MydbConfig(AppConfig):
    name = 'mydb'
```





```bash
#DB 변경사항 반영
python manage.py makemigrations myapp		#변경사항 찾기 (이력관리)
python manage.py migrate					#실제로 DB 만드는 작업
```

- class

user와 상속받고싶은 부모를 써





```bash
python manage.py shell
from myapp.models import User
datas = User.objects.all()
#객체 상속받아옴 #result
#<QuerySet [<User: kim / 김유신 / 50>]>
```



```python
u = User(userid='lee', name='임꺽정', age=40, hobby='봉사')
u.save()
```

객체생성=레코드생성

저장을 해줘야 해 = save함수 호출



## Django - Jupyter notebook 연동

```bash
#downgrade and installation
pip install django==2.0
pip install django-extensions
```

settings.py 의 `INSTALLED_APPS = ['django_extensions', ...]` 추가

```bash
#실행
python manage.py shell_plus --notebook
```



#### Model Manager

- .objects
- 데이터베이스 질의 인터페이스를 제공
- 디폴트 manager로서 모델클래스 .objects가 제공된다.
- model manager를 통해 해당 모델 클래스의 DB 데이터를 추가, 조회, 수정, 삭제가 가능하다.

#### QuerySet

- SQL을 생성해주는 인터펭시ㅡ
- `.objects.all()` : 객체 상속

- `objects.create()` : 객체생성+세이브 .실제 insert 함수

- Chaining을 지원
  
- `.objects.filter(title_icontains='1').filter(title_endswith='3')`
  
- connection 모듈을 통해 queryset으로 만들어진 실제 sql문을 shell에서 확인

  ```python
  from django.db import connection
  ModelCls.objects.all().order_by('-id')[:10]
  connection.queries[-1]
  ```


#### Select / Filtering

```python
data = 모델클래스명.objects.all()
data = data.filter(조건필트1=조건값1, ...)
data.filter(age__lte=50)	#less than or equal
```

```python
data.filter(name__icontains='김')		#ignore 대소문자
data.filter(name__contains='김')			#대소문자 구분
```

- or 조건

```python
#from django.db.models import Q
data.filter(  Q(age__gte=50) | Q(name__contains='유')  )
```

- get은 1개일때만 가능 (그 외는 예외 발생)

```python
model_instance = data.get(title='my title')
```

#### Select / Ordering

```python
data = data.order_by('field1') # 지정 필드 오름차순 요청
data = data.order_by('-field1') # 지정 필드 내림차순 요청
data = data.order_by('field2', 'field3') # 1차기준, 2차기준
```

#### Insert

- 필수필드 모두 지정해야한다. IntegrityError 발생
- blank=True, null=True, 디폴트값이 지정된 필드는 제외
- Model instance의 save함수로 저장
- Model manager의 create 함수로 저장
- 1:n 관계 생성

```python
model_instance = 모델클래스명(author=User.objects.all()[0], title='title', text='content')
model_instance.save()

new_post = 모델클래스명.objects.create(author=User.objects.get(id=1), title='title',
text='content')
```

#### Update

- Model Instance 속성을 변경하고, save 함수를 통해 저장

```python
post_instance = 모델클래스명.objects.get(id=66)
post_instance.title = 'edit title' # title 수정
post_instance.save()
```

- QuerySet의 update 함수에 업데이트할 속성값을 지정하여 일괄 수정

```python
data = 모델클래스명.objects.all()
data.update(title='test title') # 일괄 update 요청
```

#### Delete

- Model Instance 속성을 변경하고, save 함수를 통해 저장

```python
post_instance = 모델클래스명.objects.get(id=66)
post_instance.delete()
```

- QuerySet의 delete 함수에 업데이트할 속성값을 지정하여 일괄 삭제

```python
data = 모델클래스명.objects.all()
data.delete() # 일괄 delete 요청
```



### Online DB 구현

- myapp/views.py

```python
def listUser(request) :
    if request.method == 'GET' :
        delid = request.GET.get('userid', '')
        if delid != '':
            User.objects.all().get(userid=delid).delete()           #GET은 결과값이 하나일 때만 가능
            #User.objects.all().filter(userid=delid)[0].delete()    #필터는 결과값이 여러개도 가능
            return redirect('/listuser')

        data = User.objects.all()
        q = request.GET.get('q','')         #default값 설정을 위해 .get() 사용. 자기 자신 호출.
        if q != "":
            data = data.filter(name__contains=q)

        return render(request, 'template2.html', {'data':data})



    else :
        userid = request.POST['userid']
        name = request.POST['name']
        age = request.POST['age']
        hobby = request.POST['hobby']
        u = User(userid=userid, name=name, age=age, hobby=hobby)
        u.save()
        #User.objects.create()
        return redirect('/listuser')
```

template로 넘겨주는 게 좋다, for문을 사용할 수 있으므로.



- template2.html

```html
<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>

<h2>User List </h2>

{% for d in data %}
    이름 {{d.name}}   age {{d.age}}<br>
{% endfor %}

<form action="listuser">
    <input type="text" name="q">
    <input type="submit" value="검색">
    <input type="button" id="add" value="+">
</form>

<div id="layer">
    <form action="listuser" method="post">
        {% csrf_token %}
        userid: <input type="text" name="userid">
        name: <input type="text" name="name">
        age: <input type="text" name="age">
        hobby: <input type="text" name="hobby">
        <input type="submit" value="Add">
    </form>
</div>

<script>
    $("#layer").hide()				//처음에 안 보이게
    $("#add").click( function() {
        $("#layer").toggle()
    });
</script>
```

검색은 GET method, 추가는 POST method

하나의 UI에서 get/post method 다르면 구분할 수 있다.

form을 띄워주는 것은 GET 방식, form을 처리하는 것은 POST 방식으로 하여 url 경로명을 줄여나갈 수 있다. (보편적 trick)

