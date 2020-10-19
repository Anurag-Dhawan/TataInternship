from django.db import models

# Create your models here.
class Input(models.Model):
    name = models.CharField(max_length = 100)
    email =models.CharField(max_length = 100)
    company_name =models.CharField(max_length = 100)
    phone = models.IntegerField()
    text = models.TextField()

class positive(models.Model):
    name = models.CharField(max_length = 100)
    email =models.CharField(max_length = 100)
    company_name =models.CharField(max_length = 100)
    phone = models.IntegerField()
    text = models.TextField()

class negative(models.Model):
    name = models.CharField(max_length = 100)
    email =models.CharField(max_length = 100)
    company_name =models.CharField(max_length = 100)
    phone = models.IntegerField()
    text = models.TextField()

class neutral(models.Model):
    name = models.CharField(max_length = 100)
    email =models.CharField(max_length = 100)
    company_name =models.CharField(max_length = 100)
    phone = models.IntegerField()
    text = models.TextField()