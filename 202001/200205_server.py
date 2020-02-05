from flask import Flask

app = Flask(__name__)

@app.route('/')         #데이터를 네트웍 통해서 나를 호출한 곳으로 다시 보냄.
def home():
    return "hellosss^^"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)      #debug option - 파일 변경 시 자동으로 reloading

