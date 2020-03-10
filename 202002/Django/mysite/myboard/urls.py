from django.urls import path
from . import views     #from 다음에는 폴더명. import 다음에는 함수명이나
from django.shortcuts import redirect

urlpatterns = [
    path('ajaxdel', views.ajaxdel),
    path('ajaxget', views.ajaxget),
    path('login/', views.LoginView.as_view(), name='login_board'),
    path('<category>/<int:pk>/<mode>/', views.BoardView.as_view(), name='myboard'),
    #path('', lambda request: redirect('myboard', 'common', 0, 'list')),
]