from django.db import models
from django.utils import timezone

# Create your models here.

class Post(models.Model):
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    #auth.User:system table     #cascade:종속성. user 지울 때 관련된 post 자동으로 지워라.
    title = models.CharField(max_length=200)
    text = models.TextField()  # 글자수에 제한 없는 텍스트
    created_date = models.DateTimeField(            #생성된 날짜
        default=timezone.now)  # 날짜와 시간
    published_date = models.DateTimeField(
        blank=True, null=True) #  필드가 폼에서 빈 채로 저장되는 것을 허용

    def publish(self):
        self.published_date = timezone.now()        #수정된 날짜
        self.save()

    def __str__(self):      #print ftn에 적용할 str ftn
        return self.title