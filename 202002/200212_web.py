from flask import Flask, render_template, request
app = Flask(__name__)

#DB
lista = ["book2.jpg", "dog.jpg", "single.jpeg"]
listb = ['책데이터', '개영상테스트', '사람']
# 여러 리스트를 딕셔너리로 합치기
listData = []
ids = []
for i in range(len(lista)):
    ids.append(int(i))
    listData.append({'id': ids[i], 'img': lista[i], 'title': listb[i]})


@app.route('/')
def index():
    return render_template('home.html', title="my home page")

@app.route('/image')
def image():
    return render_template('image.html', listData=listData)

@app.route('/view')
def view():
    id = request.args.get('id')
    return render_template('view.html', s=listData[int(id)])

@app.route('/fileUpload', methods=['POST'])
def fileUpload() :
    if request.method == 'POST' :                  #파일 업로드는 POST 방식만 가능
        f =request.files['file1']                  #form에서의 변수
        f.save('./static/' + f.filename)  #서버의 실제 물리적인 폴더 경로
        title = request.form.get('title')

        id = len(listData)
        listData.append({'id':id, 'img':f.filename, 'title':title})
    return f"{f.filename} 파일을 {title} 제목으로 업로드 성공! <img src=/static/{f.filename}> <a href='/image'>뒤로가기</a>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
