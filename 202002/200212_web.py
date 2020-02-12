from flask import Flask, render_template, request
app = Flask(__name__)

#DB
lista = ["book2.jpg", "dog.jpg", "single.jpeg"]
listb = ['책데이터', '개영상테스트', '사람']
# 여러 리스트를 딕셔너리로 합치기
listData = []
id = []
for i in range(len(lista)):
    id.append(int(i))
    listData.append({'id': id[i], 'img': lista[i], 'title': listb[i]})


@app.route('/')
def index():
    return render_template('home.html', title="my home page")

@app.route('/image')
def image():
    return render_template('image.html', listData=listData)

@app.route('/view')
def view():
    localid = request.args.get('id')
    return render_template('view.html', s=listData[int(localid)])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)