from flask import Flask, request, jsonify
import urllib, requests, json
from bs4 import BeautifulSoup

app = Flask(__name__)

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


@app.route('/counter')
def counter():
    a = 0
    a = a + 1
    return f"{a}명이 방문했습니다."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)