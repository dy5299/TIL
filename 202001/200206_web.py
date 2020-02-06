from flask import Flask, request, jsonify
import urllib, requests, json
from bs4 import BeautifulSoup

app = Flask(__name__)

cnt = 0

@app.route('/')
def home():
    html = """
    <h1>Hello</h1>
    <img src=/static/yellow.jpg></img>
    <br>
    <iframe
        allow="microphone;"
        width="350"
        height="330"
        src="https://console.dialogflow.com/api-client/demo/embedded/rabbit5">
    </iframe>
    """
    return html    #동적 html 생성한 것임



#practice - 동적으로 HTML 만들어내는 의미
@app.route('/counter')
def counter():
    global cnt
    cnt += 1
    
    cnt_str = str(cnt)
    output = ""
    for i in cnt_str:
        temp = f"<img src=/static/{i}.png width=64>"
        output = str(output) + str(temp)
    return output + "명이 방문했습니다."

#list 방법으로 반복
@app.route('/counter_pf_1')
def counter_pf_1():
    global cnt
    cnt += 1

    html = "".join(  [  f"<img src=/static/{i}.png width=32>" for i in str(cnt)  ]  )
    html += "명이 방문했습니다."
    return html

#for 방법으로 반복
@app.route('/counter_pf_2')
def counter_pf_2():
    global cnt
    cnt += 1
    
    html=""
    for i in str(cnt) :
        html += f"<img src=/static/{i}.png width=32>"
    html += "명이 방문했습니다."
    return html





@app.route('/weather')
def weather():
    city = request.args.get('city')
    return f"{city} 날씨 좋아요"












if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)