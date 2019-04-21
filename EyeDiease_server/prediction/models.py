from django.db import models

# Create your models here.
class Eye(models.Model): 
    name = models.CharField(max_length=50) 
    Patient_Eye_Img = models.ImageField(upload_to='images/') 
