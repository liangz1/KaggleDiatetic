from random import random
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .forms import *
import os
from .predict import *
from .Inception import InceptionDR
# Create your views here.
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
best_model_path='/home/ubuntu/EyeDiease_server/prediction/inception_v3_50_50_13_best_f1.h5'
global best_model
best_model = InceptionDR("eval")
best_model.load_best_model(best_model_path)
global graph
graph = tf.get_default_graph()

@csrf_exempt
def index(request):
   # best_model_path='/home/ubuntu/EyeDiease_server/prediction/inception_v3_0.h5'
    #best_model = InceptionDR("eval")
    #best_model.load_best_model(best_model_path)

    if request.method == 'POST':
        print (request)
        print (request.POST)
        print (request.FILES)
        form = EyeImageForm(request.POST, request.FILES)

        if form.is_valid():
            result = form.save()
            #return JsonResponse({"data":1.666})
            print (os.getcwd() + result.Patient_Eye_Img.url)
            with graph.as_default():
                ret = predict(os.getcwd() + result.Patient_Eye_Img.url, best_model)
            print (ret)
            #return HttpResponse(ret)
            return JsonResponse({'data':str(ret)})
            #return HttpResponse(str(random()))
            #return redirect('success') 
    else:
        form = EyeImageForm()
    return render(request, 'index.html', {'form' : form})


def success(request):
    return HttpResponse('successfuly uploaded')
