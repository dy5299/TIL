from flask import Flask, escape, request

app = Flask(__name__)

# set FLASK_APP=파일명.py
# flask run

@app.route('/')
def hello():
    name = request.args.get("name","World")
    return f'Hello, {escape(name)}!'

@app.route('/hi', methods=['POST'])
def hi():
    return {
        "version" : "2.0",
        "templete" : {
            "outputs" : [
                {
                    "simpleText":{
                        "text" : "간단한 텍스트 요소입니다."
                    }
                }
            ]
        }
    }



if __name__ == "__main__":              
    app.run(host="127.0.0.1", port="5000")