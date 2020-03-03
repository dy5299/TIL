from django.urls import path
from . import views     #from 다음에는 폴더명. import 다음에는 함수명이나

urlpatterns = [
    path('', views.index),
#    path('<name>/', views.index2),      #<name> parameter에 대해 동적으로 mapping
#    path('<int:pk>/detail', views.index3),

    path('login/', views.LoginView.as_view(), name='login'),    # class base

    path('list/', views.list, name='list'),
    path('<int:pk>/detail/', views.detail, name='detail'),       #function base
    path('add/', views.PostView.as_view(), name='add'),
    path('<int:pk>/edit/', views.PostEditView.as_view(), name='edit'),
]