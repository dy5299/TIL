from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
#from blog.models import Post       #폴더의존성 줄이기위해
from . import models
from django.views.generic import View
from django.contrib.auth import authenticate
from django.contrib.auth.models import User

#다른 py에서 호출
#from blog.forms import PostForm     #함수 바로 호출 가능
from . import forms                #forms.함수명 으로 함수 호출 가능

# Create your views here.

def index(request):
    return HttpResponse("INDEX. OKOKOK")

def index2(request, name):
    return HttpResponse("INDEX2 OK" + name)

def index3(request, pk):
    #p = models.Post.objects.get(pk=pk)         #왼pk: parameter name, 오pk: variable 같은 것을 찾아주는 함수(get)
    p = get_object_or_404(Post, pk=pk)      #에러나면 아래 return이 아닌, pagenotfound(404) exception로 리턴시킨다.
    return HttpResponse("INDEX3 OK" + p.title)

def list(request):
    username = request.session['username']              #text
    user = User.objects.get(username=username)          #object
    data = models.Post.objects.all().filter(author=user)
    context = {"data":data, 'username':username}
    return render(request, "blog/list.html", context)

def detail(request, pk):
    p = get_object_or_404(models.Post, pk=pk)      #에러나면 아래 return이 아닌, pagenotfound(404) exception로 리턴시킨다.
    return render(request, "blog/detail.html", {"d":p})



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


'''
class PostView(View):
    def get(self, request):
        username = request.session['username']
        return render(request, "blog/add.html", {'username':username})

    def post(self, request):
        title = request.POST.get('title')
        text = request.POST.get('text')
        username = request.session['username']
        user = User.objects.get(username=username)
        models.Post.objects.create(title=title, text=text, author=user)    #create:생성과 동시에 save
        return redirect('list')
'''



class PostEditView(View):
    def get(self, request, pk):     #특정 포스트를 수정하므로 pk parameter를 받아와야 한다.
        #초기값 지정
        if pk == 0 :
            form = forms.PostForm()       #empty form
        else :
            post = get_object_or_404(Post, pk=pk)
            #form = forms.PostForm(initial={'title':post.title, 'text':post.text})     #초기값을 post form으로 채움
            #post와 form을 강제로 mapping시킴.
            form = forms.PostForm(instance=post)      #instance라는 parameter에 model data(post) 넣음
        return render(request, "blog/edit.html", {'form':form})

    def post(self, request, pk):

        username = request.session["username"]
        user = User.objects.get(username=username)

        if pk == 0:
                #models.Post.objects.create(title=form['title'].value(), text=form['text'].value(), author=user)
                #form과 model data를 강제로 mapping
                form = forms.PostForm(request.POST)       #받은 데이터로 폼 채움
        else:
                post = get_object_or_404(Post, pk=pk)
                #post.title = form['title'].value()  #form과 model data를 강제로 mapping
                #post.text = form['text'].value()
                form = forms.PostForm(request.POST, instance=post)

        if form.is_valid():
            post = form.save(commit=False)
            if pk == 0 :
                post.author = user
                post.save()         #form data로부터 post data(model data)를 얻기 위해서 save. NOT for save.
            else :
                post.publish()
            return redirect('list')
        return render(request, 'blog/edit.html', {'form':form})


