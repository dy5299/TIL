from flask import Flask, escape, request
import pickle

app = Flask(__name__)

# set FLASK_APP=파일명.py
# flask run
#with open('data.pickle','rb') as f
#db = pickle.load()

db = pickle.load('./db.bin')
db={
    "0" : {
        "name" : "bill",
        "phone" : "010-1230-1231"
    },
    "1" : {
        "name" : "bab",
        "phone" : "010-5566-1231"
    },
    "2" : {
        "name" : "test",
        "phone" : "010-7979-1231"
    }
}  #dictionary
id=0

@app.route('/users', methods=['POST'])
def create_user():
    #body = ...
    # todo body에 id를 넣어준다.
    global id
    body = request.json
    print(body)
    body['id'] = int(id)
    db[str(id)]=body['id']
    db = pickle.dump(db, './db.bin')  #db가 바뀔 때마다 dump
    id+=1
    return body

@app.route('/users/{id}', methods=['GET'])
def select_user(id):
    if id not in db:
        return {}, 404
    print(db)
    return db[id]

@app.route('/users/{id}', methods=['DELETE'])
def delete_user(id):
    del db[id]
    db = pickle.dump(db, './db.bin')
    return {}

@app.route('/users/{id}', methods=['PUT'])
def update_user(id):
    db[name].append(body[name])
    db[phone].append(body[phone])
    return {}

@app.route('/hi', methods=['POST'])
def hi():
    return {
        "version" : "2.0",
        "templete" : {
            "outputs" : [
                {
                    "simpleText":{
                        "text" : "반가워~ (발그레)"
                    }
                }
            ]
        }
    }



if __name__ == "__main__":              
    app.run(host="127.0.0.1", port="5000")