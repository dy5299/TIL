#-*- coding: utf-8 -*-
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
    a={
        "version" : "2.0",
        "templete" : {
            "outputs" : [
                {
                    "simpleText":{
                        "text" : "반가워~(발그레)"
                    }
                }
            ]
        }
    }
    print(a)
    return a


@app.route('/namecard', method=['POST'])
def namecard():
    body = request.json
    image_url = body['userRequest']['utterance']

    if image_url startswitch 'http//':
        save()

    print('image_url>' + image_url)
    # f = request.files['file]
    # f.save(secure_filename(f.filename))
    
    return {
        "version" : "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText":{
                        "text": "명함정보는 ~~~입니다"
                    }
                }
            ]
        }
    }




#if __name__ == "__main__":              
#    app.run(host="127.0.0.1", port="5000")