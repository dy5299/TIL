from django.urls import path
from . import views     #from 다음에는 폴더명. import 다음에는 함수명이나

urlpatterns = [
    path('', views.index),
    path('test/main', views.test),
    path('login', views.login),
    path('logout', views.logout),
    path('service', views.service),
    path('uploadimage', views.uploadimage),
]