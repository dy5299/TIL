from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import face

def index(request):
    return HttpResponse("Hello DJango!!!^^^^")


def test(request):
    data = {"s":{"img":"test.png"}, 
        "list":[1,2,3,4,5]  }   
    return render(request, 'myapp/template.html', data) 
    
def login(request):
    id = request.GET["id"]    
    pwd = request.GET["pwd"]
    if id == pwd :         
        request.session["user"] = id
        return redirect("/service")
    return redirect("/static/login.html")

def logout(request):
    request.session["user"] = ""
    #request.session.pop("user")
    return redirect("/static/login.html")
    
    
def service(req):  
    if  req.session.get("user", "") == "" :
        return redirect("/static/login.html") 
    html = "Main Service<br>"  + req.session.get("user") + "감사합니다<a href=/logout>logout</a>"
    return HttpResponse(html)

@csrf_exempt
def uploadimage(request):   

    file = request.FILES['file1']
    filename = file._name    
    fp = open(settings.BASE_DIR + "/static/" + filename, "wb")
    for chunk in file.chunks() :
        fp.write(chunk)
    fp.close()
    
    result = face.facerecognition(settings.BASE_DIR + "/known.bin", settings.BASE_DIR + "/static/" + filename)
    if result != "" : 
        request.session["user"] = result    
        return redirect("/service")
    return redirect("/static/login.html")