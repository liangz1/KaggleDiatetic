# forms.py 
from django import forms 
from .models import *
  
class EyeImageForm(forms.ModelForm): 
  
    class Meta: 
        model = Eye 
        fields = ['name', 'Patient_Eye_Img'] 
