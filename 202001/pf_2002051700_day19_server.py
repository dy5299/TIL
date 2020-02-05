from flask import Flask, request
import requests
import urllib
from bs4 import BeautifulSoup

def getWeather(city) :    
    url = "https://search.naver.com/search.naver?query="
    url = url + urllib.parse.quote_plus(city + "날씨")
    print(url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    temp = bs.select('span.todaytemp')    
    desc = bs.select('p.cast_txt')    
    return  {"temp":temp[0].text, "desc":desc[0].text}            

app = Flask(__name__)

@app.route('/')
def home():    
    name = request.args.get("name")    
    item = request.args.get("item")    
    return "hello--^^^^---" + name + item


@app.route('/abc')
def abc():
    return "test~~~~~~~"

@app.route('/weather')
def weather():
    city = request.args.get("city")
    info = getWeather(city)
    
    return  "<font color=red>" + info["temp"] + "도   "  + info["desc"]  + "</font>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)