from django.db import models
from django.utils import timezone

class Board(models.Model):
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    text = models.TextField()  # 글자수에 제한 없는 텍스트
    created_date = models.DateTimeField(            #생성된 날짜
        default=timezone.now)  # 날짜와 시간
    published_date = models.DateTimeField(
        blank=True, null=True) #  필드가 폼에서 빈 채로 저장되는 것을 허용
    cnt = models.IntegerField(default=0)            #조회수
    image = models.ImageField(null=True, blank=True)     #이미지는 필수 아님
    category = models.CharField(max_length=10, default='common')

    def __str__(self):
        return self.title

    def publish(self):
        self.published_date = timezone.now()        #수정된 날짜
        self.save()

