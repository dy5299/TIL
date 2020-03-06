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
        return redirect('0/list')




class PostEditView(View):
    def get(self, request, pk, mode):     #특정 포스트를 수정하므로 pk parameter를 받아와야 한다.
        if mode == 'list':
            username = request.session['username']  # text
            user = User.objects.get(username=username)  # object
            data = models.Post.objects.all().filter(author=user)
            context = {"data": data, 'username': username}
            return render(request, "blog/list.html", context)
        elif mode == 'detail':
            p = get_object_or_404(models.Post, pk=pk)  # 에러나면 아래 return이 아닌, pagenotfound(404) exception로 리턴시킨다.
            return render(request, "blog/detail.html", {"d": p})
        elif mode == 'add' :
            form = forms.PostForm()       #empty form
            return render(request, "blog/edit.html", {'form': form})
        elif mode == 'edit' :
            post = get_object_or_404(models.Post, pk=pk)
            form = forms.PostForm(instance=post)      #instance라는 parameter에 model data(post) 넣음
            return render(request, "blog/edit.html", {'form':form})
        else :
            return HttpResponse("error page")

    def post(self, request, pk, mode='edit'):

        username = request.session["username"]
        user = User.objects.get(username=username)

        if pk == 0:
                form = forms.PostForm(request.POST)       #받은 데이터로 폼 채움
        else:
                post = get_object_or_404(models.Post, pk=pk)
                form = forms.PostForm(request.POST, instance=post)

        if form.is_valid():
            post = form.save(commit=False)
            if pk == 0 :
                post.author = user
                post.save()         #form data로부터 post data(model data)를 얻기 위해서 save. NOT for save.
            else :
                post.publish()
            return redirect('board', 0, 'list')
        return render(request, 'blog/edit.html', {'form':form})


