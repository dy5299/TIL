from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


def index(request):
    return HttpResponse("WE ARE NEW APP, AJAX!!!!!!!!!")

#단순하게 form을 띄우는 역할
def calcform(request):
    return render(request, "ajax/calc.html")    # 경로는 template/ajax/calc.html

#실제 계산
def calc(request):
    op1 = int(request.GET["op1"])
    op2 = int(request.GET["op2"])
    result = op1 + op2
    #return HttpResponse("{'result':" + str(result) + "}")
    return JsonResponse({'error':0, 'result':result})      #dictionary to json 변환



#로그인 기능

def loginform(request):
    return render(request, "ajax/login.html")

def login(request):
    id = request.GET['id']
    pwd = request.GET['pwd']
    if id == pwd:
        request.session['user'] = id
        return JsonResponse({'error':0})        #error code를 만들어두면 어느 페이지로 갈지 여기서는 신경쓸 필요 없다.
    return JsonResponse({'error':-1, 'message':'아이디와 패스워드를 다시 확인해주세요.'})



#업로드
def uploadform(request):
    return render(request, "ajax/upload.html")


#@csrf_exempt   내가 책임질테니 예외처리 해 (체크하지 말아줘)
def upload(request):
    file = request.FILES['file1']
    filename = file._name
    fp = open(settings.BASE_DIR + "/static/" + filename, "wb")
    for chunk in file.chunks():
        fp.write(chunk)
    fp.close()
    return HttpResponse("UPLOAD")
