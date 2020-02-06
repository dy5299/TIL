from flask import Flask, request, jsonify
import urllib, requests, json
import math
from bs4 import BeautifulSoup
import pickle

def getWeather(city) :
    url = 'https://search.naver.com/search.naver?query='
    url = url + urllib.parse.quote_plus(city + " 날씨")
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    temp = bs.select('span.todaytemp')
    desc = bs.select('p.cast_txt')
    return {"temp":temp[0].text, "desc":desc[0].text}

def getQuery(word) :
    url = 'https://search.naver.com/search.naver?where=kdic&query='
    url = url + urllib.parse.quote_plus(word)
    
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    
    output = bs.select('p.txt_box')
    return [node.text for node in output]



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')         #데이터를 네트웍 통해서 나를 호출한 곳으로 다시 보냄.
#decorator: 함수 호출 시 함수 앞뒤로 코드 삽입된다.
def home():
    name = request.args.get("name")     #각 변수명은 상관없지만 통상적으로 일치시킨다.
    item = request.args.get("item")
    #return "hellosss~~~~~~" + name + item
    return "hi"


@app.route('/weather')
def weather():
    city = request.args.get("city")
    info = getWeather(city)
    #return "<font color=red>" + info["temp"] + "도 / " + info["desc"] + "</font>"
    #기본적으로 HTML 형식으로 전달된다.
    #return json.dumps(info)
    #json.dumps(source) : JSON 형식으로 변환한다.
    return jsonify(info)



@app.route('/dialogflow', methods=['GET', 'POST'])   #두 방식 모두 동작한다. GET 방식을 추가하면 Debugging할 때 편리하다.
def dialogflow():
    req = request.get_json(force=True)      #강제로 받은 데이터를 JSON format으로 변환
    print(json.dumps(req, indent=4))

    answer = req['queryResult']['fulfillmentMessages']
    intentName = req['queryResult']['intent']['displayName']
    
    if intentName == 'query' :
        word = req["queryResult"]['parameters']['any']
        text = getQuery(word)[0]
        res = {'fulfillmentText':text}
        
    elif intentName == 'weather' :
    #elif intentName == 'weather' and req['queryResult']['allRequiredParamsPresent'] == True :
    #뒤 조건은 안써도 된다. dialogflow에서 required parameter를 먼저 받고 서버로 보내준다.
        date = req["queryResult"]['parameters']['date']
        geo_city= req["queryResult"]['parameters']['geo-city']
        
        info = getWeather(geo_city)
        text = f"{date}의 {geo_city} 날씨정보 : {info['temp']} ℃ / {info['desc']}"
        res = {'fulfillmentText': text}
        
    elif intentName == 'orderfood2' :
        price = {"짜장면":5000, "짬뽕":10000,"탕수육":20000}
        params = req['queryResult']['parameters']['food_number']
    
        output = [  food.get("number-integer", 1)*price[food["food"]] for food in params  ]
        text = f"총 금액은 {math.floor(sum(output))}원입니다."
        res = {'fulfillmentText': text}

    else :
        res = {'fulfillmentText':answer}
    
    return jsonify(res)





def processDialog(req) :
    
    answer = req['queryResult']['fulfillmentText']
    intentName = req['queryResult']['intent']['displayName'] 
    
    if intentName == 'query' :
        word = req["queryResult"]['parameters']['any'] 
        text = getQuery(word)[0]                
        res = {'fulfillmentText': text}   
    else :
        res = {'fulfillmentText': answer}
        
    return res


@app.route('/dialogflow_unittest', methods=['GET',"POST"])
def dialogflow_unittest():

    if request.method == 'GET' :
        file = "static/200206_json.json"
        with open(file, encoding='UTF8') as json_file:
            req = json.load(json_file)
            print(json.dumps(req, indent=4, ensure_ascii=False))


    else :
        req = request.get_json(force=True)
        print(json.dumps(req, indent=4, ensure_ascii=False))
    return jsonify(processDialog(req))





@app.route('/s5/<name>')
def s6(name):
    return "data=" + name

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)      #debug option - 파일 변경 시 자동으로 reloading
#위 코드 뒤에 있는 @app.route는 실행이 안 되네..l,