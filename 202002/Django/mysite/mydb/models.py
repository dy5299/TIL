from django.db import models

# Create your models here.
class User2(models.Model) :
    userid = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=10)
    age = models.IntegerField()
    hobby = models.CharField(max_length=20)

    def __str__(self):                          #print 적용할 때 자동으로 적용되는 함수
        return f"{self.userid} / {self.name} / {self.age}"
        return f"{self.userid} / {self.name} / {self.age}"