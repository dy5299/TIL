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

## 챗봇 생성 실습

Create agent - 챗봇 하나. 이름은 영어로 써야 코딩 시 편리

Intents - training : 어순, 시간, 장소 등은 알아서 학습.

integrations - Web Demo abled



Responses 에서 parameter 받아오기 : `$food`

부모 context의 parameter 가져오기 : `#orderfood-custom-followup.food`

부모가 아닌 context 끌고 오기 : input context 에 참조할 context 추가 후 `#order_food-followup.name`