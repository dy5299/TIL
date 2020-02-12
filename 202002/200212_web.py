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


def goURL(msg, url) :
    html = f"""
<script>
    alert("@msg")
    window.location.href = "@url"
</script>
    """ #위의 html은 단지 문자열일 뿐이다. 서버에서는 문자열을 리턴하고, 브라우저에서 html을 디코딩한다.
    html = html.replace("@msg", msg)
    html = html.replace("@url", url)
    return html


@app.route('/')
def index():
    return render_template('home.html', title="my home page")

@app.route('/image')
def image():
    return render_template('image.html', listData=listData)

@app.route('/view')     #/view?id=0
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
    return goURL("업로드가 성공했습니다.","/image")

@app.route('/delete')   #/delete?id=0
def delete() :
    id = request.args.get('id')
    del listData[int(id)]
    return goURL("데이터를 삭제하였습니다.","/image")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
