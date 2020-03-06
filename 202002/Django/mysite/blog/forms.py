from . import models
from django.forms import Form, CharField, Textarea, ValidationError
from django import forms

def validator(value):
    if len(value) < 5 : raise ValidationError("길이가 너무 짧아요");


class PostForm(forms.ModelForm):
    #    title = CharField(label='제목', max_length=20, validators=[validator])
    #    text = CharField(label='내용', widget=Textarea)
    class Meta:
        model = models.Post    #model은 쟝고가 정의한 것. 에다 model data를 넣어준다
        fields = ['title', 'text']      #가져올 fields만 선택 가능하다.

    def __init__(self, *args, **kwargs):                    #내 생성자 정의
        super(PostForm, self).__init__(*args, **kwargs)     #부모 생성자 호출
        self.fields['title'].validators = [validator]

