from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.generic import View
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.core.files.uploadedfile import SimpleUploadedFile

from . import models
from . import forms
from . import apps

# Create your views here.
from mysite import settings


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

            page = request.GET.get('page', 1)
            p = Paginator(data, 3)  # collection 형태의 데이터면 상관 없다 / page 당 개수
            sub = p.page(page)

            context = {"datas": sub, 'username': username, 'category':category}
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

        if mode == 'add':
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


'''
# paging unit test
def page(request):
    datas = [{'id': 1, 'name': '홍길동1'},
             {'id': 2, 'name': '홍길동2'},
             {'id': 3, 'name': '홍길동3'},
             {'id': 4, 'name': '홍길동4'},
             {'id': 5, 'name': '홍길동5'},
             {'id': 6, 'name': '홍길동6'},
             {'id': 7, 'name': '홍길동7'}, ]  # list
    page = request.GET.get('page',1)
    p = Paginator(datas,3)      #collection 형태의 데이터면 상관 없다 / page 당 개수
    sub = p.page(page)

    return render(request, apps.APP + "/page.html", {'datas': sub})
'''

def ajaxdel(request):
    pk = request.GET.get('pk')
    board = models.Board.objects.get(pk=pk)
    board.delete()
    return JsonResponse({'error':'0'})

def ajaxget(request):
    username = request.session["username"]
    user = User.objects.get(username=username)


    page = request.GET.get('page',1)
    datas = models.Board.objects.all().filter(author=user, category='common')

    page = int(page)
    subs = datas[(page-1)*3:(page)*3]
    '''p = Paginator(datas, 3)
    subs = p.page(page)'''

    datas = {'datas': [{'pk': sub.pk, 'title': sub.title, 'cnt': sub.cnt} for sub in subs]}
    return JsonResponse(datas, json_dumps_params={'ensure_ascii':False})
    #마지막 JsonResponse 뒤에 옵션은, json만 출력했을 때 한글이 제대로 보이도록 하는 옵션이다.
    #자바스크립트가 알아서 인코딩하므로 없어도 상관 없다.


from django.db import connection
def dictfetchall(cursor):
    desc = cursor.description
    return [
        dict(zip([col[0] for col in desc], row))
        for row in cursor.fetchall()
    ]

def listsql(request, category, page):
    username = request.session["username"]

    # data
    cursor = connection.cursor()
    cursor.execute(f"""
    select b.id, title, cnt, username
    from myboard_board b, auth_user u
    where b.author_id = u.id and username='{username}' and category='{category}'
    """)
    datas = dictfetchall(cursor)

    page = int(page)
    subs = datas[(page - 1) * 3:(page) * 3]
    context = {'datas': subs}

    return render(request, 'myboard/mylistsql.html', context)

def gallery(request):
    username = request.session["username"]

    #data
    cursor = connection.cursor()
    sql = f"""
    select filename
    from myboard_image
    where author_id = ( select id from auth_user where username='{username}' )
    """
    cursor.execute(sql)

    data = dictfetchall(cursor)
    context = {'data': data, 'username': username}

    return render(request, 'myboard/gallery.html', context)

def upload(request):
    username = request.POST['username']

    #file upload
    # f = open(filename, 'rb')
    # file_upload = SimpleUploadedFile(filename, f.read(), content_type='image/jpeg')

    #file save
    file = request.FILES['filename']
    filename = file._name
    print(filename)
    fp = open(settings.BASE_DIR + "/static/faces/" + username + '/' + filename, "wb")
    for chunk in file.chunks():
        fp.write(chunk)
    fp.close()

    #db insert
    sql = f"""
    INSERT INTO myboard_image (author_id, filename) VALUES (
    (SELECT id from auth_user where username='{username}'), '{filename}');
    """
    cursor = connection.cursor()
    cursor.execute(sql)
    data = dictfetchall(cursor)
    #return render(request, "myboard/gallery.html", )
    return redirect('/myboard/gallery')
