from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse
from blog.models import Post
from django.views.generic import View
from django.contrib.auth import authenticate
# Create your views here.


def index(request) :
    return HttpResponse("ok")

def index2(request, name) :
    return HttpResponse("ok " + name)

def index3(request, pk) :
    #p = Post.objects.get(pk=pk)


    p = get_object_or_404(Post, pk=pk)
    return HttpResponse("ok " + p.title)

def list(request) :

    username = request.session["username"]

    data = Post.objects.all()
    context = {"data":data, "username":username}
    return render(request, "blog/list.html", context)

def detail(request, pk) :
    p = get_object_or_404(Post, pk=pk)
    return render(request, "blog/detail.html", {"d":p})



class PostView(View) :
    def get(self, request):
        return HttpResponse("get ok~")

    def post(self, request):
        return HttpResponse("post ok~")


class LoginView(View) :
    def get(self, request):
        return render(request, "blog/login.html")

    def post(self, request):
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(username=username, password=password)
        if user == None :
            return redirect("login")
        request.session["username"] = username
        return redirect("list")