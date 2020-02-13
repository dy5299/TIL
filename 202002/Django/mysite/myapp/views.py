from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import face


def index(request):
    return HttpResponse("Main Django!!!")


def test(request):
    out = {'s': {'img':'test.png'},     #안에서 딕셔너리 정의
           'list': [1,2,3,4,5]          #안에서 리스트 정의
           }
    return render(request, 'template.html', out)
    #쟝고에서는 딕셔너리로만 데이터 전달 가능.


#원래는 정의해주는게 맞아. 보안적인 측면에서 login을 static에 넣는 건 bad
'''def login(request):
    return render(request, 'login.html')'''

def login(request):
    id = request.GET['id']
    pwd = request.GET['pwd']
    if id == pwd:
        request.session['user'] = id
        return redirect('/service')
    return redirect('/static/login.html')


def logout(request):
    request.session['user']=''
    request.session.pop('user')
    return redirect('/static/login.html')


def service(request):
    if request.session.get('user','') == '':
        return redirect('/static/login.html')

    html = "Main Service<br>" + request.session.get('user') + "님 로그인되었습니다.<br><a href=/logout>로그아웃</a>"
    return HttpResponse(html)

@csrf_exempt
def uploadimage(req):
    file = req.FILES['file1']
    filename = file._name
    fp = open(settings.BASE_DIR + '/static/'+ filename, 'wb')
    for chunk in file.chunks():
        fp.write(chunk)
    fp.close()

    print(settings.BASE_DIR)

    result = face.faceverifiation(settings.BASE_DIR + '/known.bin', settings.BASE_DIR + "/static/" + filename)[0]
    print(result)
    if result != "" :
        req.session['user'] = result
        return redirect('/service')
    if file =="":
        return HttpResponse("파일을 업로드해주세요.")
    return redirect('/static/login.html')