# 챗봇



## AI 스피커

### 특장점 비교

아마존(echo):	처음으로 ai 스피커 만들었다.

소비자들은 인공지능 기능이 필요해서 ai 스피커를 산 건 아니다. 최소한 무선 스피커 기능을 하기 때문에 구매한 것이다.

알람이 성공 요인인 것으로 분석됐다. 가장 많이 쓰는 기능이다.

실제로 비서 역할보다는(ai 서비스) 무선 스피커 기능으로 주로 소비되는 것 같다.

네이버:			검색 우위. 자연어 처리 잘 함

구글:				ㄱ

카카오(다음):	상대적으로 자연어 처리는 떨어짐



### 국내 AI 스피커 시장 전망

AI 스피커 보급률은 PC 수준까지 늘 것

KT, SKT는 통신사로 인공지능 플랫폼 공급 우위

향후 서비스 제공자인 네이버, 카카오가 시장 과점 예상

LG, 삼성같은 제조사가 들어올 수 있는 영역은 아님. 가격 경쟁력이 없다.



phone gateway - 구글 AI 듀플렉스 : 전화를 대신 걸어주는 AI



## 챗봇의 장점 웹사이트에 비해

- 챗봇은 자연어 기반으로 하기 때문에, 새로운 유저 인터페이스 학습이 필요없다.
- 웹사이트는 1:N인데 비해, 챗봇은 1:1 mapping된다. 개인화된 유저 경험에 기반하여 제공된다.
- re-engage with users in a relevant way
- 웹사이트는 사이트맵(tree map)에서 찾아가야 하지만, 챗보은 더 나은, 더 빠른 경험을 제공한다.
- handoff(담당 부서가 아닙니다-다시 설명해야 함). 해결할 수 없는 부분이 사람에게 넘어가도 내역이 있으니 중복해서 상황을 설명할 필요가 없다

챗봇 수는 150k to 2430k 수준까지 증가 예상. 개인화 시대니까. TV방송과 유투브의 차이도 마찬가지 아닐까

인공지능 스피커 마이크는 특정 방향만 인식하도록 되어있다. (지향성 마이크, 마이크가 여러 개 내장되어 있다)

# Dialogflow

챗봇 대부분은 알고리즘 보다는 데이터 추가 형식으로...

## 구성요소 설명

구성요소: Intent, Entities, Training, Integrations, Fulfillment, Event

### intent

화자의 의도인 intent 는 실제 대화를 의미합니다. 그리고 화자의 의도를 파악한다는 것은 지금 챗봇에 들어온 대화를 기반으로 Dialogflow 의 다양한 intent 중에서 사용자가 말한 intent 를 파악하는 것으로 intent matching 이라고 합니다. 다시 말하면 intent 는 사용자가 말하는 것과 Dialogflow 가 수행해야 할 작업 간의 매핑을 나타내는 것입니다.

Fallback intent 는 말 그대로 사용자의 대화가 어떤 intent 와도 매칭되지 않을때 선택되어지는 intent 이며(‘잘 못 알아들었습니다.’ 나 ‘다시 한번 말씀해주세요’ 등) 좀 더 나은 머신러닝을 위한 training 을 할 때 nagative example 을 넣을 수도 있어서 잘못 intent 가 매핑되는 것을 막는 용도로도 활용할 수 있습니다.

### entity

‘내일 오후 2시 되나요’ 라는 사용자의 질의에서 중요한 속성 항목인 ‘내일’, ‘오후 2시’ 를 파라미터로 뽑아 내는 것을 Entity 라고 합니다.

### context

‘내일 오후 2시 되나요’ 에서 무엇을 위한 ‘내일 오후 2시’ 인가를 파악하기 위해서 전체 대화의 문맥을 사람이 이해하는 것처럼, 그 전에 대화가 되었던 ‘수리’ 라는 것을 기억하는 것을 의미합니다.

### fullfillment

명시한 서버를 거쳐와서 답변하는 방식이다.

### integrations

다양한 플랫폼과 연동하는 기능이다.

### event

문맥을 jump할 수 있는 기능

## 챗봇 실습

- agent 생성

Create agent - 챗봇 하나 단위이다. 이름은 영어로 써야 코딩 시 편리하다.

Intents - training : 어순, 시간, 장소 등은 구글이 알아서 학습한다.

integrations - Web Demo abled 웹 테스트 용이

- parameter 이용

Responses 에서 parameter 받아오기 : `$food`

부모 context의 parameter 가져오기 : `#orderfood-custom-followup.food`

부모가 아닌 context 끌고 오기 : input context 에 참조할 context 추가 후 `#order_food-followup.name`

- 복합 entity (pairing)

synonyms 체크 해제

`@food:food` 앞: 다른 entity 이름, 뒤: 파라미터명

`@food:food @sys.number-integer:number-integer ` : 복합 entities

parameter 의 IS LIST 속성을 ON시키면 다음과 같은 복합 entities, 리스트 형태로 반환 가능하다. JSON 형태로 반환된다.

```bash
[ { "food": "짜장면", "number-integer": 2 }, { "number-integer": 2, "food": "짬뽕" } ]
```

- 숫자 생략 시

짜장면, 짬뽕 2개

```bash
[ { "food": "짜장면" }, { "food": "짬뽕", "number-integer": 2 } ]
```

### python 연동 - openAPI

python 연동에는 두 가지 방법 있다. openAPI 이용하여 갖다 쓰는 방법과, fullfillment 통해서 - 필요한 logic만 집어넣는 방식

get 방식은 보안 문제가 있다. 주소에 query string 담겨있다. 브라우저에서 호출할 수 있다.

post 방식은 브라우저에서 호출할 수 없다.

코딩으로는 둘 다 호출 가능하다.

```python
import requests
import json

def get_answer(text, sessionId):
    data_send = {
        'query': text, 'sessionId': sessionId,
        'lang': 'ko', 'timezone' : 'Asia/Seoul'
    }
    data_header = {
        'Authorization': 'Bearer 420208382d5046eb89a9e1fa3e31e4cb',
        'Content-Type': 'application/json; charset=utf-8'
    }

    dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'
    res = requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)
    if res.status_code == requests.codes.ok:
        return res.json()    
    return {}

dict = get_answer("부산 내일 날씨 어때", 'user01')
answer = dict['result']['fulfillment']['speech']
print("Bot:" + answer)
```

- requests.post

`requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)`

html 주소, content(name value 상의 데이터):일반적인 문자형태, header:dictionary 형태

```python
res = requests.get("http://www.naver.com")
print(res)
print(type(res))    #객체
print(res.text)
```

```bash
<Response [200]>
<class 'requests.models.Response'>
~html내용~
```

- JSON print

```python
data_send = {
    'query': '부산 날씨 어때', 
    'sessionId': 'user01',
    'lang': 'ko', 
    'timezone' : 'Asia/Seoul'
}
data_header = {
    'Authorization': 'Bearer 420208382d5046eb89a9e1fa3e31e4cb',
    'Content-Type': 'application/json; charset=utf-8'
}

dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'
res1 = requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)
    
print(res1.text)
```

```json
{
  "id": "beedc765-1063-43d9-b9fe-9c9ed4926898-ce609cdc",
  "lang": "ko",
  "sessionId": "user01",
  "timestamp": "2020-02-04T07:39:51.96Z",
  "result": {
    "source": "agent",
    "resolvedQuery": "부산 날씨 어때",
    "action": "",
    "actionIncomplete": true,
    "score": 0.57886386,
    "parameters": {
      "geo-city": "부산광역시",
      "date": ""
    },
    "contexts": [
      {
        "name": "d9f7e7fa-f1f4-4e72-82fc-d3cc87a5c050_id_dialog_context",
        "lifespan": 2,
        "parameters": {
          "geo-city": "부산광역시",
          "geo-city.original": "부산",
          "date": "",
          "date.original": ""
        }
      },
      {
        "name": "weather_dialog_context",
        "lifespan": 2,
        "parameters": {
          "geo-city": "부산광역시",
          "geo-city.original": "부산",
          "date": "",
          "date.original": ""
        }
      },
      {
        "name": "weather_dialog_params_date",
        "lifespan": 1,
        "parameters": {
          "geo-city": "부산광역시",
          "geo-city.original": "부산",
          "date": "",
          "date.original": ""
        }
      },
      {
        "name": "__system_counters__",
        "lifespan": 1,
        "parameters": {
          "no-input": 0.0,
          "no-match": 0.0,
          "geo-city": "부산광역시",
          "geo-city.original": "부산",
          "date": "",
          "date.original": ""
        }
      }
    ],
    "metadata": {
      "intentId": "d9f7e7fa-f1f4-4e72-82fc-d3cc87a5c050",
      "intentName": "weather",
      "webhookUsed": "false",
      "webhookForSlotFillingUsed": "false",
      "isFallbackIntent": "false"
    },
    "fulfillment": {
      "speech": "날짜가 빠졌습니다",
      "messages": [
        {
          "type": 0,
          "speech": "날짜가 빠졌습니다"
        }
      ]
    }
  },
  "status": {
    "code": 200,
    "errorType": "success"
  }
}
```

- 필요한 값 출력해보기

```python
data_send = {
    'query': '부산 날씨 어때', 
    'sessionId': 'user01',
    'lang': 'ko', 
    'timezone' : 'Asia/Seoul'
}
data_header = {
    'Authorization': 'Bearer 420208382d5046eb89a9e1fa3e31e4cb',
    'Content-Type': 'application/json; charset=utf-8'
}

dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'
res1 = requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)

result = res1.json()
print('intentName: ', result['result']['metadata']['intentName'])
print('actionIncomplete: ', result['result']['actionIncomplete'])

params = result['result']['parameters']
for p in params:
    print(p, params[p])
```

```bash
#'query'='부산 날씨 어때'
intentName:  weather
actionIncomplete:  True
geo-city 부산광역시
date 
---------------------------
#'query'='오늘 부산 날씨 어때'
intentName:  weather
actionIncomplete:  False
geo-city 부산광역시
date 2020-02-04
---------------------------
#'query'='안녕'
intentName:  Default Welcome Intent
actionIncomplete:  False
---------------------------
#'query'='짜장면 2, 짬뽕 5'
intentName:  orderfood2
actionIncomplete:  False
food_number [{'food': '짜장면', 'number-integer': 2.0}, {'food': '짬뽕', 'number-integer': 5.0}]
```

반환값이 list [       안에 dictionary {}           ]

- 총 금액 계산

```python
#간단히
import math
costs = {"짜장면":5000, "짬뽕":10000,"탕수육":20000}

while True :
    txt = input("->")
    dict = get_answer(txt, 'user01')
    answer = dict['result']['fulfillment']['speech']
    if dict['result']['metadata']['intentName'] == 'orderfood2' and dict['result']['actionIncomplete'] == False :
        foodnum = dict['result']['parameters']['food_number']

        totalcost = 0
        for i in range(len(foodnum)) :
            totalcost = totalcost + foodnum[i]['number-integer'] * costs[foodnum[i]['food']]
        totalcost = math.floor(totalcost)

        print("총 금액은 ", totalcost, "원 입니다.")
    else :
        print("Bot:" + answer)
```



(200205)

# UI/UX

UI는 인터페이스, 즉 정보기기나 소프트웨어의 화면 등 사람과 접하는 면을 설계하는 일이다.
반면 **UX(사용자 경험)란 특정 정황과 목표를 갖는, 정보기기/서비스 사용자의 "느낌,태도,행동"을 말한다.**

UI에서 U가 보편적 인간을 모델로 한 분석 대상이었다면, UX에서 U는 주관적 인간을 모델로 한 공감 대상이다.

출처: https://story.pxd.co.kr/567 [pxd UX Lab.]



제주도를 갈 수 있는 방법?

UI : 가장 효율적인 방법을 찾는 것. 비행기

UX : 상황에 따라 다를거야. 몇 시간 걸리더라도 아이유와 같이 가는 것. ~ apple 성공 요인

# Practice - Fibonacci Sequence

## recursive function

했던 계산을 여러번 하기 때문에 시간이 오래 소요된다.

```python
def fibo(n):
    if n<3 :
        return 1
    else :
        return fibo(n-1) + fibo(n-2)

print(fibo(45))	#1134903170		2분정도 걸림
```

## dictionary 이용

캐시 메모리 컨셉으로 dictionary 를 사용할 수 있다.

이미 계산된 것을 또 계산하지 않는다.

```python
pre = {1:1, 2:1}
def fibo2(n) :
    v = pre.get(n)
    if v != None :
        return v
    else :
        sum = fibo2(n-1) + fibo2(n-2)
        pre[n] = sum
        return sum
```

- using list

```python
def fibo(a1,a2,n) :
    out = [a1,a2]
    for i in range(1,n+1):
        out.append(out[i]+out[i-1])

    return out[n-1]

print(fibo(1,1,40))
```

# 웹 정보 이용

## WEB - HTTP Method (GET/POST)

웹서버에 클라이언트의 data를 전송하기 위한 전송방법에는 GET 과 POST가 있다.

GET 방식은 URL을 통해 data를 전송하고, POST 방식은 body를 통해 data를 전송한다.

GET 방식은 전송 URL이 노출되어 보안에 취약하고 길이 제한이 있다.

POST 방식은 길이 제한이 없다.

## in Python

- 파이썬 표준 패키지(beautifulsoup)를 이용

- 한글이 url 코드로 인코딩 해야한다.

https://search.naver.com/search.naver?query=%EB%B6%80%EC%82%B0%EA%B4%91%EC%97%AD%EC%8B%9C+%EB%82%A0%EC%94%A8

### HTML Parsing

```python
import urllib, requests, json
from bs4 import BeautifulSoup

#1st method
res = requests.get(url)
html = res.text
print(html)

#2nd method
html = urllib.request.urlopen(url).read()
print(html)

#3rd method - urllib.request 혹은 requests
bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
bs = BeautifulSoup(requests.get(url).text, "html.parser")
print(bs.html)
```

방법 1과 2는 한글이 코드로 나올 수도 있다.

반면 방법3처럼 BeautifulSoup 을 거치면 무조건 한글이 인코딩되어 출력된다.

### selecting elements

```python
city = '부산광역시'
url = 'https://search.naver.com/search.naver?query='
url = url + urllib.parse.quote_plus(city + " 날씨")


bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
```

#### findAll

```python
temp = bs.findAll("span", "todaytemp")
desc = bs.findAll("p", "cast_txt")
```

findAll : BeautifulSoup 에서만 제공하는 비표준 방식으로 (태그, attribute;default:클래스) 이다. 문법 체계가 없다.

#### CSS selector

```python
temp = bs.select('span.todaytemp')
desc = bs.select('p.cast_txt')
```

select : CSS selector 라는 문법 체계가 있다. (DB의 SQL과 비슷한 컨셉)

앞에 아무것도 없으면 태그, `.`은 클래스, `#`은 id, `nth-child(2)` 은 두 번째 자식 노드를 가리킨다.

- selector 문법 예시 (full directory)

```bash
#main_pack > div.sc.cs_weather._weather > div:nth-child(2) > div.weather_box > div.weather_area._mainArea > div.today_area._mainTabContent > div.main_info > div > p > span.todaytemp
```

### 함수로 정의

```python
#함수로 정의
def getWeather(city) :
    url = 'https://search.naver.com/search.naver?query='
    url = url + urllib.parse.quote_plus(city + " 날씨")
    #url 요청
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    #select
    temp = bs.select('span.todaytemp')
    desc = bs.select('p.cast_txt')
    
    return temp[0].text + '℃ / ' + desc[0].text

getWeather('제주도')
#result
'1℃ / 흐림, 어제보다 4˚ 낮아요'
```

문자열로 리턴하는 것은 좋은 방법이 아니다. 함수 밖에서 가공할 수 없기 때문이다.

함수 내에서 딕셔너리로 반환하면 추후 정보 가공이 용이하다.

```python
return temp[0].text + '℃ / ' + desc[0].text
```

### 최종

```python
#pf 최종
def get_answer(text, sessionId):
    data_send = {
        'query': text, 'sessionId': sessionId,
        'lang': 'ko', 'timezone' : 'Asia/Seoul'
    }
    data_header = {
        'Authorization': 'Bearer 420208382d5046eb89a9e1fa3e31e4cb',
        'Content-Type': 'application/json; charset=utf-8'
    }

    dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'
    res = requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)
    
    if res.status_code == requests.codes.ok:
        return res.json()    
    return {}


def getWeather(city) :
    url = 'https://search.naver.com/search.naver?query='
    url = url + urllib.parse.quote_plus(city + " 날씨")
    
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    
    temp = bs.select('span.todaytemp')
    desc = bs.select('p.cast_txt')
    
    return {"temp":temp[0].text, "desc":desc[0].text}


price = {"짜장면":5000, "짬뽕":10000,"탕수육":20000}

while True :
    txt = input("->")
    dict = get_answer(txt, 'user01')
    answer = dict['result']['fulfillment']['speech']
    
    
    if dict['result']['metadata']['intentName'] == 'orderfood2' :
        params = dict['result']['parameters']['food_number']
    
        output = [  food.get("number-integer", 1)*price[food["food"]] for food in params  ]
        print(sum(output))
        
    elif dict['result']['metadata']['intentName'] == 'weather' and dict['result']['actionIncomplete'] == False :
        date = dict['result']['parameters']['date']
        geo_city= dict['result']['parameters']['geo-city']
        
        info = getWeather(geo_city)
        
        print(f"{date}의 {geo_city} 날씨정보 : {info['temp']} ℃ / {info['desc']}")
        
    else :
        print("Bot:" + answer)
```

```bash
#result
->오늘 부산 날씨
2020-02-05의 부산광역시 날씨정보 : 3 ℃ / 맑음, 어제보다 6˚ 낮아요

->오늘 날씨
Bot:어느 도시의 날씨를 알고 싶으신가요?
->서울
2020-02-05의 서울특별시 날씨정보 : -6 ℃ / 맑음, 어제보다 10˚ 낮아요

->제주도 날씨
Bot:날짜가 빠졌습니다
->내일
Bot:어느 도시의 날씨를 알고 싶으신가요?
->제주도
Bot:도시를 알려주세요
->제주
2020-02-06의 제주시 날씨정보 : 3 ℃ / 흐림, 어제보다 4˚ 낮아요
```

### 지식백과 결과

```python
def getQuery(word) :
    url = 'https://search.naver.com/search.naver?where=kdic&query='
    url = url + urllib.parse.quote_plus(word)
    
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    
    output = bs.select('p.txt_box')
    
    return [node.text for node in output]
```

리턴값을 리스트 형태로 반환한다.