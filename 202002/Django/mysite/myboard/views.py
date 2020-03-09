from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.views.generic import View
from django.contrib.auth import authenticate
from django.contrib.auth.models import User

from . import models
from . import forms
from . import apps

# Create your views here.

def index(request):
    return HttpResponse("INDEX. OKOKOK")



class LoginView(View):
    def get(self, request):
        return render(request, apps.APP+"/login.html")

    def post(self, request):
        #Loging 처리
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user == None :
            return redirect('login_board')    #urls.py에서 지정한 name. NOT 경로명.

        #로그인 성공한 경우
        request.session['username'] = username
        return redirect(apps.APP+'all/0/list')




class BoardView(View):
    def get(self, request, pk, mode, category):     #특정 포스트를 수정하므로 pk parameter를 받아와야 한다.
        if mode == 'list':
            username = request.session['username']  # text
            user = User.objects.get(username=username)  # object
            if category == 'all' :
                data = models.Board.objects.all().filter(author=user)
            else :
                data = models.Board.objects.all().filter(author=user, category=category)
            context = {"data": data, 'username': username, 'category':category}
            return render(request, apps.APP+"/list.html", context)

        elif mode == 'detail':
            p = get_object_or_404(models.Board, pk=pk)  # 에러나면 아래 return이 아닌, pagenotfound(404) exception로 리턴시킨다.
            p.cnt += 1
            p.save()
            return render(request, apps.APP+"/detail.html", {"d": p, 'category':category})

        elif mode == 'add' :
            form = forms.BoardForm(initial={'category':category})       #empty form
            return render(request, apps.APP+"/edit.html", {'form': form, 'category':category})

        elif mode == 'edit' :
            post = get_object_or_404(models.Board, pk=pk)
            form = forms.BoardForm(instance=post)      #instance라는 parameter에 model data(post) 넣음
            return render(request, apps.APP+"/edit.html", {'form':form, 'category':category})

        else :
            return HttpResponse("error page")

    def post(self, request, pk, mode='edit', category='common'):
        username = request.session["username"]
        user = User.objects.get(username=username)

        if pk == 0:
                form = forms.BoardForm(request.POST)       #받은 데이터로 폼 채움
        else:
                post = get_object_or_404(models.Board, pk=pk)
                form = forms.BoardForm(request.POST, instance=post)

        if form.is_valid():
            post = form.save(commit=False)
            if pk == 0 :
                post.author = user
                post.save()         #form data로부터 post data(model data)를 얻기 위해서 save. NOT for save.
            else :
                post.publish()
            return redirect('myboard', category, 0, 'list')
        return render(request, apps.APP+'/edit.html', {'form':form})
