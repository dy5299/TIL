from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from blog.models import Post
from django.views.generic import View
from django.contrib.auth.models import User
from django.contrib.auth import authenticate

# Create your views here.

def index(request):
    return HttpResponse("INDEX. OKOKOK")

def index2(request, name):
    return HttpResponse("INDEX2 OK" + name)

def index3(request, pk):
    #p = Post.objects.get(pk=pk)         #왼pk: parameter name, 오pk: variable 같은 것을 찾아주는 함수(get)
    p = get_object_or_404(Post, pk=pk)      #에러나면 아래 return이 아닌, pagenotfound(404) exception로 리턴시킨다.
    return HttpResponse("INDEX3 OK" + p.title)

def list(request):
    username = request.session['username']              #text
    user = User.objects.get(username=username)          #object
    data = Post.objects.all().filter(author=user)
    context = {"data":data, 'username':username}
    return render(request, "blog/list.html", context)

def detail(request, pk):
    p = get_object_or_404(Post, pk=pk)      #에러나면 아래 return이 아닌, pagenotfound(404) exception로 리턴시킨다.
    return render(request, "blog/detail.html", {"d":p})


class PostView(View):
    def get(self, request):
        return render(request, "blog/edit.html")

    def post(self, request):
        title = request.POST.get('title')
        text = request.POST.get('text')
        username = request.session['username']
        user = User.objects.get(username=username)
        Post.objects.create(title=title, text=text, author=user)    #create:생성과 동시에 save

        return redirect('list')


class LoginView(View):
    def get(self, request):
        return render(request, "blog/login.html")

    def post(self, request):
        #Loging 처리
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user == None :
            return redirect('login')    #urls.py에서 지정한 name. NOT 경로명.

        #로그인 성공한 경우
        request.session['username'] = username
        return redirect('list')
