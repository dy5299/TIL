from flask import Flask, request
import urllib, requests, json
from bs4 import BeautifulSoup

def getWeather(city) :
    url = 'https://search.naver.com/search.naver?query='
    url = url + urllib.parse.quote_plus(city + " 날씨")
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    temp = bs.select('span.todaytemp')
    desc = bs.select('p.cast_txt')
    return {"temp":temp[0].text, "desc":desc[0].text}




app = Flask(__name__)

@app.route('/')         #데이터를 네트웍 통해서 나를 호출한 곳으로 다시 보냄.
#decorator: 함수 호출 시 함수 앞뒤로 코드 삽입된다.
def home():
    name = request.args.get("name")     #각 변수명은 상관없지만 통상적으로 일치시킨다.
    item = request.args.get("item")
    return "hellosss^^-----" + name + item


@app.route('/weather')
def weather():
    city = request.args.get("city")
    info = getWeather(city)
    return "<font color=red>" + info["temp"] + "도 / " + info["desc"] + "</font>"
    #기본적으로 HTML 형식으로 전달된다.


@app.route('/abc')
def home2():
    return "abcabc^^"

@app.route('/s5/<name>')
def s6(name):
    return "data=" + name

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)      #debug option - 파일 변경 시 자동으로 reloading
#위 코드 뒤에 있는 @app.route는 실행이 안 되네..